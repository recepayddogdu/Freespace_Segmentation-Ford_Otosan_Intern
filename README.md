# Freespace Segmentation with Fully Convolutional Neural Network (FCNN)
Drivable area detection in highway images with semantic segmentation.

The project was developed with technologies such as **Python**, **Pytorch**, **opencv**.

Below are the image used while testing the model, the expected output from the model, and the drivable fields that the model predicts;

<p  align="center">
<img  src="images/predict_wlogo.gif"  width="">
</p> 

**[Click here](/Questions_and_Answers.md) for the question and answer page that includes basic information about semantic segmentation and deep learning.**

## Parts of the project;
- [Json to Mask](#json-to-mask)
- [Mask on Image](#mask-on-image)
- [Preprocess](#preprocess)
- [Model](#model)
- [Train](#train)
- [Data Augmentation](#data-augmentation)
- [Predict](#predict)

## Json to Mask
JSON files are obtained as a result of highway images tagged by Ford Otosan Annotation Team. The JSON files contain the exterior and interior point locations of the freespace (drivable area) class. 

The file structure is as follows;

<p  align="center">
<img  src="images/json_file.png"  width="">
</p> 

A mask was created with the data in the JSON file to identify the pixels with freespace in the image.

The `fillPoly` function from the cv2 library was used to draw the masks.

    for obj in json_dict["objects"]:
        if obj["classTitle"]=="Freespace":
            cv2.fillPoly(mask, np.array([obj["points"]["exterior"]], dtype=np.int32), color=100)
            
            if obj["points"]["interior"] != []:
                for interior in obj["points"]["interior"]:
                    cv2.fillPoly(mask, np.array([interior], dtype=np.int32), color=0)

Mask example;
<p  align="center">
<img  src="images/mask_array.png"  width="">
</p> 

<p  align="center">
<img  src="images/json2mask.png"  width="">
</p> 

**Click for the codes of this section; [json2mask.py](src/json2mask.py)**

## Mask on Image

In this section, masks obtained from JSON files were added on the images and tested whether the masks were correct.

Adding on the image by coloring the pixels designated as freespace and with 50% opacity:

    image[mask==100, :] = (255, 0, 125)
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)

Mask on image example:

<p  align="center">
<img  src="images/maskonimage.png"  width="">
</p>

**Click for the codes of this section; [mask_on_image.py](src/mask_on_image.py)**

## Preprocess

The images and masks refer to "features" and "labels" for Segmentation. To feed them into the Segmentation model, which will be written in PyTorch, we need to format them appropriately.

**Image Normalization** is a process in which we change the range of pixel intensity values to make the image more familiar or normal to the senses.

	   img = cv2.imread(image_path)
	   zeros_img = np.zeros((1920, 1208))
	   norm_img = cv2.normalize(img, zeros_img, 0, 255, cv2.NORM_MINMAX)

Pytorch inputs must be in Tensor format.
Image to Tensor;

	    torch_image = torch.from_numpy(image_array).float()

The same procedures are applied to masks. 

In addition, one hot encoding was done. One Hot Encoding means binary representation of categorical variables. This process first requires the categorical values to be mapped to integer values. Then, each integer value is represented as a binary vector with all values zero except the integer index marked with 1.

<p  align="center">
<img  src="images/one-hot.jpg"  width="">
</p>


    def one_hot_encoder(res_mask,n_classes):
		one_hot=np.zeros((res_mask.shape[0],res_mask.shape[1],n_classes),dtype=np.int)
		for i,unique_value in enumerate(np.unique(res_mask)):
			one_hot[:,:,i][res_mask==unique_value]=1
		return one_hot

One Hot Encoder output;

<p  align="center">
<img  src="images/onehotencoder.png"  width="">
</p>

**Click for the codes of this section; [preprocess.py](src/preprocess.py)**

## Model

The U-Net model was used in the project. Because it is clear that one of the models that give the best results for semantic segmentation is U-Net. Semantic segmentation, also known as pixel-based classification, is an important task in which we classify each pixel of an image as belonging to a particular class. U-net is a encoder-decoder type network architecture for image segmentation. U-net has proven to be very powerful segmentation tool in scenarios with limited data. The ability of U-net to work with very little data and no specific requirement on input image size make it a strong candidate for image segmentation tasks.

<p  align="center">
<img  src="images/u-net.png"  width="">
</p>

- The **encoder** is the first half in the architecture diagram. Apply convolution blocks followed by a maxpool downsampling to encode the input image into feature representations at multiple different levels.

		self.maxpool = nn.MaxPool2d(2)

<p  align="center">
<img  src="images/maxpool.gif"  width="">
</p>

- The **decoder** is the second half of the architecture. The goal is to semantically project the discriminative features (lower resolution) learnt by the encoder onto the pixel space (higher resolution) to get a dense classification. The decoder consists of **upsampling** and **concatenation** followed by regular convolution operations.

		self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
		 --
		 x = torch.cat([x, conv2], dim=1)

<p  align="center">
<img  src="images/upsampling.gif"  width="">
</p>

#### Activation Function
**ReLu** was used as the activation function.

- The  **rectified linear activation function**  or  **ReLU**  for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance.

	    nn.ReLu(inplace=True)

<p  align="center">
<img  src="images/relu.png"  width="">
</p>

#### Output Layer

		x = nn.Softmax(dim=1)(x)
	
The **softmax** function is a function that turns a vector of K real values into a vector of K real values that sum to 1. The input values can be positive, negative, zero, or greater than one, but the softmax transforms them into values between 0 and 1, so that they can be interpreted as probabilities. If one of the inputs is small or negative, the softmax turns it into a small probability, and if an input is large, then it turns it into a large probability, but it will always remain between 0 and 1.

<p  align="center">
<img  src="images/softmax.png"  width="">
</p>

**Click for the codes of this section; [UNet_1.py](src/UNet_1.py)**

## Train

## Data Augmentation

## Predict


