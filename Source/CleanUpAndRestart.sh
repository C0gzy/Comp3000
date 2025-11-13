# if argument All is passed, then clean up and restart the model
if [ "$1" == "All" ]; then
    rm -rf ../CoralDataSetAugmented
fi

mv coral_bleaching_model.h5 ./ModelHistory/V$(date +%Y%m%d_%H%M%S).h5
rm best_coral_model.h5

mv training_history.png ../ProjectInfomation/Graphs/V$(date +%Y%m%d_%H%M%S).png