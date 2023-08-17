rm [!0]*
rm [0][!012]*
echo "Previous run deleted"

source "$HOME"/.bashrc
source "$ACCOUNT_PATH"/load-dakota.sh
echo "DAKOTA sourced"

echo "Running DAKOTA..."
dakota --input 01_dakota_input.in --output 08_final_results.out --error 09_error.err