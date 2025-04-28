find . -iname "*.tar" -exec mkdir "{}.dir" \; ; find . -iname "*.tar" -exec tar -C "{}.dir" -xf {} \;

find . -iname "*.pdf" -exec pdfcrop --margins '0 0 0 0' {} \;


