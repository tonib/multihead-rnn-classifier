# https://stackoverflow.com/questions/1885525/how-do-i-prompt-a-user-for-confirmation-in-bash-script
read -p "Are you sure you want to delete the model and data cache? (y/n)" -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm -rf data/cache
    mkdir data/cache
    rm -rf model
    echo
    echo "Model and cache deleted"
fi
