TRAINING_DATA_URL="paultimothymooney/chest-xray-pneumonia"
NOW=$(date)

kaggle datasets download -d $TRAINING_DATA_URL -p ds_project/ds_project_package/image_classification_model/datasets && \
unzip ds_project/ds_project_package/image_classification_model/datasets/chest-xray-pneumonia.zip -d ds_project/ds_project_package/image_classification_model/datasets && \
echo $TRAINING_DATA_URL 'retrieved on:' $NOW > ds_project/ds_project_package/image_classification_model/datasets/training_data_reference.txt