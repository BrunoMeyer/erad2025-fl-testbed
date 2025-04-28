import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    # Argparse
    parser = argparse.ArgumentParser(description='Plot F1-Score and Communication Time')
    parser.add_argument('--dir', type=str, help='Directory containing client metrics')
    parser.add_argument('--figftype', type=str, help='Figure file type', default='pdf', choices=['pdf', 'png'])


    parser.add_argument('--fwidth', type=int, help='Figure width', default=12)
    parser.add_argument('--fheight', type=int, help='Figure height', default=4)

    # Bool if savefig
    parser.add_argument('--savefig', action='store_true', help='Save figure as pdf')

    args = parser.parse_args()

    base_dir = args.dir

    # List directories in the current directory (ignore files)
    directories = [(d, f'{base_dir}{d}') for d in os.listdir(base_dir) if os.path.isdir(f'{base_dir}/{d}')]
    fig, ax1 = plt.subplots(figsize=(args.fwidth, args.fheight))

    # Set font size for everything, including axis labels, ticks and title
    FONTSIZE = 18
    plt.rcParams.update({'font.size': FONTSIZE})

    ax2 = ax1.twinx()

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    # markers = ['o', 's', 'D', '^', 'v', '<', '>']
    markers = [None, '.', 'o', 's', 'D', '^', 'v', '<', '>']

    for d_label, d in sorted(directories, key=lambda x: x[0]):
        print(d)
        # Find in the directory all files that matches client_*_metrics.csv

        df_list = []
        files = []
        server_metrics_path = None
        for root, _, filenames in os.walk(d):
            for filename in filenames:
                if filename.startswith("client_") and filename.endswith("_metrics.csv"):
                    files.append(os.path.join(root, filename))
                elif filename == "server_metrics.csv":
                    server_metrics_path = os.path.join(root, filename)
        
        df_list = [pd.read_csv(f) for f in files]
        server_metrics = pd.read_csv(server_metrics_path)
        
        max_len = max([len(df) for df in df_list])
        f1_list = []
        train_time_list = []
        eval_time_list = []
        auc_list = []
        total_time_list = []
        for df_c in df_list:
            print(len(df_c))
            # right_zero_padding_size = max_len - len(df_list[i])
            # df_list[i] = pd.concat([df_list[i], pd.DataFrame([[None]*len(df_list[i].columns)]*right_zero_padding_size, columns=df_list[i].columns)], ignore_index=True)
            # df_c = pd.concat([pd.DataFrame([[None]*len(df_c.columns)], columns=df_c.columns), df_c], ignore_index=True)

            df_f1 = df_c[['F1-Score']]
            df_train_time = df_c[['Training Time (s)']]
            df_eval_time = df_c[['Evaluation Time (s)']]
            df_auc = df_c[['AUC']]
            df_total_time = df_c[['Total Time (s)']]

            f1_list.append(df_f1)
            train_time_list.append(df_train_time)
            eval_time_list.append(df_eval_time)
            auc_list.append(df_auc)
            total_time_list.append(df_total_time)

        # Average the metrics for each round
        f1_list = pd.concat(f1_list, axis=1).mean(axis=1)
        train_time_list = pd.concat(train_time_list, axis=1).mean(axis=1)
        eval_time_list = pd.concat(eval_time_list, axis=1).mean(axis=1)
        auc_list = pd.concat(auc_list, axis=1).mean(axis=1)
        total_time_list = pd.concat(total_time_list, axis=1).mean(axis=1)
        
        # Add an empty line on the beggining of the server metrics
        # server_metrics = pd.concat([pd.DataFrame([[None]*len(server_metrics.columns)], columns=server_metrics.columns), server_metrics], ignore_index=True)
        # server_metrics = server_metrics[:len(f1_list)]
        # print(server_metrics)
        
        round_idx = server_metrics['Round']
        time_list = server_metrics['Time Since Start (s)']
        aggregation_time = server_metrics['Aggregation Time (s)']
        print(len(f1_list), len(time_list), len(aggregation_time), len(total_time_list))

        comm_time = [
            (time_list[i] - (time_list[i-1]) if i > 0 else 0) - (aggregation_time[i]+total_time_list[i])
            for i in range(1, len(f1_list))
        ]

        f1_list = f1_list[1:]
        total_time_list = total_time_list[1:]
        time_list = time_list[1:len(f1_list)+1]
        aggregation_time = time_list[1:len(f1_list)+1]

        print(len(f1_list), len(time_list), len(aggregation_time), len(total_time_list), len(comm_time))

        # Plot using index as x axis
        # fig, ax = plt.subplots(1, 1)
        # plt.show()

        color = colors.pop(0)
        
        ax1.plot(time_list, f1_list, color=color, marker=markers[0], linestyle='-', label=f'{d_label} F1-Score')
        ax2.plot(time_list, comm_time, color=color, marker=markers[1], linestyle='--', label=f'{d_label} Communication')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(20))
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)

        ax1.set_xlabel('Time Since Start (s)', fontsize=FONTSIZE)
        # ax1.set_ylabel('Average F1-Score (macro) among clients', fontsize=FONTSIZE)
        ax1.set_ylabel('F1-Score', fontsize=FONTSIZE)
        # ax2.set_ylabel('Communication Time (s)', fontsize=FONTSIZE)
        ax2.set_ylabel('Communication (s)', fontsize=FONTSIZE)
        
        ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2

        # Resort by label name
        handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))

        ax1.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=int(FONTSIZE*0.8))

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Save fig as pdf
    plt.tight_layout()
    if args.savefig:
        # plt.savefig(f'{base_dir}f1_communication_time.{args.figftype}', format=args.figftype, bbox_inches='tight')
        figname = base_dir.replace('/', '_')
        plt.savefig(f'{figname}_f1_communication_time.{args.figftype}', format=args.figftype, bbox_inches='tight')
    else:
        plt.show()




if __name__ == "__main__":
    main()