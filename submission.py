import caffe
import cv2
import sys

def deploy(img_path):

    caffe.set_mode_gpu()
    
    # Initialize the Caffe model using the model trained in DIGITS. Which two files constitute your trained model?
    net = caffe.Classifier('/dli/data/digits/20190201-222749-1fd6/deploy.prototxt', '/dli/data/digits/20190201-222749-1fd6/snapshot_iter_540.caffemodel',
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
                       
    # Create an input that the network expects. This is different for each project, so don't worry about the exact steps, but find the dataset job directory to show you know that whatever preprocessing is done during training must also be done during deployment.
    input_image= caffe.io.load_image(img_path)
    input_image = cv2.resize(input_image, (256,256))
    mean_image = caffe.io.load_image('/dli/data/digits/20190201-221714-3580/mean.jpg')
    input_image = input_image-mean_image

    # Make prediction. What is the function and the input to the function needed to make a prediction?
    prediction = net.predict([input_image])##REPLACE WITH THE FUNCTION THAT RETURNS THE OUTPUT OF THE NETWORK##([##REPLACE WITH THE INPUT TO THE FUNCTION##])

    # Create an output that is useful to a user. What is the condition that should return "whale" vs. "not whale"?
    if prediction.argmax()==0:
        return "whale"
    else:
        return "not whale"

    
##Ignore this part    
if __name__ == '__main__':
    print(deploy(sys.argv[1]))

