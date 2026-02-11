# Submission

Source URL: https://www.codabench.org/competitions/12566/

## Submission format

### File packaging

- Translate the validation images provided. For SAR → EO, the target resolution is **256x256** pixels. For SAR → RGB, RGB → IR, and SAR→ IR, the target resolution is **1024x1024** pixels. If these target resolutions are not met, the evaluation script will resample the images to meet them. Images can be submitted in PNG or TIFF format. 

- Create a ZIP submission file containing all your translated images with 4 subdirectories named "sar2eo", "sar2rgb", "rgb2ir", "sar2ir", and the a readme.txt Note that the readme.txt should be in the root of the archive. The scoring program is looking for folders with these specific names. Failure to match the folder structure will result in failed scoring. 

 **Correct structure:**
	
 submission.zip

 └── sar2eo

 ├── 66.png

 :

 └── 441011.png

 └── sar2rgb

 ├── 0.png

 :

 └── 59.png

 └── sar2ir

 ├── 0.png

 :

 └── 59.png

 └── rgb2ir

 ├── 0.png

 :

 └── 59.png

 └── readme.txt

Generated images can be submitted in PNG or TIFF formats. The target resolutions for each track are as follows:

 - SAR → EO: 256 x 256
 - SAR → RGB: 1024 x 1024
 - RGB → IR: 1024 x 1024
 - SAR → IR: 1024 x 1024

### File naming convention

#### Results

- **SAR → EO**

 - SAR images: e.g., Gotcha_1234.png, where 1234 is the image_id - EO images corresponding to the SAR images: e.g., Gotcha_1234.png, corresponding to the SAR image with id "1234"
 - The valid/test data contains SAR images for the SAR track and both SAR and EO images for (SAR+EO) track of the challenge. The purpose is to use the provided (SAR+EO) train image to deisgn and implement a method for translating SAR images to EO images. 

- **SAR → RGB**, **RGB → IR**, **SAR → IR**

 - Contains four seperate locations with a stack of geo-spatially aligned images. The locations are UC Davis, Califonia; Manhattan, New York; Bingham Copper Mine, Utah; and Centerfield, Utah.
 - Due to variability in the sensors, the resolutions vary widely for each chip. We ask for submissions to target a 1024x1024 pixel image. We provide the raw data to allow users freedom to choose resampling methods. Feel free to email mavoc.pbvs@gmail.com with any questions or concerns you may have. 
 - RGB and IR channels are designated with <>_rgb.tiff, and <>_ir.tiff respectively. SAR images are usually designated with <>_GEC.tiff
 - We recommend the use of rasterio for loading images. Please see our boilerplate code for examples. 

#### Readme

The readme.txt file should contain the following lines filled in with the runtime per image (in seconds) of the solution, 1 or 0 accordingly if employs CPU or GPU at runtime, and 1 or 0 if employs extra data for training the models or not.

 runtime per image [s] : 10.43 
 CPU[1] / GPU[0] : 1
 Extra Data [1] / No Extra Data [0] : 1
 Other description : Solution based on A+ of Timofte et al. ACCV 2014. We have a Matlab/C++ implementation, and report single core CPU runtime. The method was trained on Train 91 of Yang et al. and BSDS 200 of the Berkeley segmentation dataset. 

The last part of the file can have any description you want about the code producing the provided results (dependencies, link, scripts, etc.)
The provided information is very important both during the validation period when different teams can compare their results / solutions but also for establishing the final ranking of the teams and their methods.

### Additional notes

- During the testing phase of the competition, teams are allowed **up to ten submissions per day**. 
- During the evaluation phase, teams submit their label predictions and credibility score to be evaluated on the competition server. 
- Teams are allowed **up to 12 submissions**, which prevents them from effectively fine-tuning on the test dataset. Results are made visible during both phases.
- Only one submission file (`submission.zip`) is allowed per submission.
- Do not include ground truth masks, scripts, or additional files in the ZIP archive.
- Participants are strongly encouraged to validate the structure and contents of their submission before uploading to the Codabench platform.
- Submissions that do not follow the specified format may lead to evaluation errors or **disqualification**.
