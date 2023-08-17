# Get the evaluation number and create a new directory for the simulation
num=$(echo "$1" | awk -F. '{print $NF}')

# Get the script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cp params.tmp.$num params.in.$num

# Generate submit.sh and the FDS input file
$ACCOUNT_PATH/miniconda3/bin/python3 ../../python-scripts/fill_templates.py "$1" "$num" "$SCRIPT_DIR"

# Run the simulation with sbatch
sleep $[ ( $RANDOM % 10 )  + 1 ]s
sbatch submit-"$num".sh &> sbatch-"$num".out
jobid=$(tail -1 sbatch-"${num}".out | grep -E -o '[0-9]+')
echo "Submitted eval $num with jobid $jobid"

# Wait until the batch job finishes before continuing.
STATUS="RUNNING"
array=("RUNNING" "PENDING" "COMPLETING" "CONFIGURING" "REQUEUED")
while [[ " ${array[*]} " =~ $STATUS ]];
  do
    sleep 60
    STATUS=$(sacct -j "$jobid" | awk 'NR==3{print $6}')
    echo "eval $num has status: $STATUS"
  done

# Write the evaluation number to the results file
echo "$num" > results.tmp.$num