#!/bin/bash
function print_help()
{
    if [[ 1 == $# ]]; then
        echo -e "\n\t\033[31m$1\033[0m"
    fi

    local usage="
    Usage: \n
    ./run.sh -h <host_name>  -g <gpu_id> -sp <save_path>
             -ne <num_epochs> -lr <learning_rate>
             -bs <batch_size> -kp <keep_prob> -beta <beta>
             -lu <lstm_units>

    host_name:
        guangzhou, wuhan, nanjing

    gpu_id:
        Which gpu/gpus to use according to your device.

    save_path:
        Model output base directory.

    num_epochs:
        Number of epochs to run trainer.

    learning_rate:
        Initial learning rate.

    batch_size:
        Batch size for training.

    keep_prob:
        The keep probability for lstm dropout.

    beta:
        The regularization term for l2 norm.

    lstm_units:
        lstm output units.

    log file: LOG

    "
    echo -e "\033[33m${usage}\033[0m"
}

py_command=''
HOST="guangzhou"

if [ $# -eq 1 ]; then
    print_help
    exit 0
fi

while [[ $# > 1 ]]
do
key="$1"

case $key in
    -h|--hostname)
    HOST="$2"
    shift # past argument
    ;;
    -g|--gpu)
    py_command=$py_command" --gpu=""$2"
    shift # past argument
    ;;
    -sp|--save_path)
    py_command=$py_command" --save_path=""$2"
    shift # past argument
    ;;
    -ne|--num_epochs)
    py_command=$py_command" --num_epochs=""$2"
    shift # past argument
    ;;
    -lr|--learning_rate)
    py_command=$py_command" --learning_rate=""$2"
    shift # past argument
    ;;
    -bs|--batch_size)
    py_command=$py_command" --batch_size=""$2"
    shift # past argument
    ;;
    -kp|--keep_prob)
    py_command=$py_command" --keep_prob=""$2"
    shift # past argument
    ;;
    -beta)
    py_command=$py_command" --beta=""$2"
    shift # past argument
    ;;
    -lu|--lstm_units)
    py_command=$py_command" --lstm_units=""$2"
    shift # past argument
    ;;
    *)
    echo "Unknown command line option:$key"
    ;;
esac
shift # past argument or value
done

echo "#!/bin/bash">qsub.sh
echo "source /aifs/users/rcz56/env/bin/activate">>qsub.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64'>>qsub.sh
echo "python scripts/train/train_normal.py"$py_command>>qsub.sh

host_list=('guangzhou' 'wuhan' 'nanjing' 'jinan')

if [ " ${host_list[@]} " =~ " ${HOST} " ] ; then
    qsub -cwd -S /bin/bash -o LOG -j y -l hostname=$HOST qsub.sh
else
    print_help
    exit 0
fi

rm qsub.sh

