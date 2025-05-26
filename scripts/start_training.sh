export PYTHONXOPTIONS='-Xfrozen_modules=off'
export PYDEVD_DISABLE_FILE_VALIDATION=1
# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Please create it first with: python -m venv venv"
    exit 1
fi

accelerate launch models/roformer/roformer_train.py ${1:-"--resume"}