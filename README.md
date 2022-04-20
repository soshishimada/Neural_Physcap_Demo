 # Demo: "Neural PhysCap" Neural Monocular 3D Human Motion Capture with Physical Awareness
The implementation is based on [SIGGRAPH '21](http://vcai.mpi-inf.mpg.de/projects/PhysAware/).   
Neural Monocular 3D Human Motion Capture with Physical Awareness  
Authors: Soshi Shimada  Vladislav Golyanik  Weipeng Xu  Patrick Pérez  Christian Theobalt  
 
## Dependencies
- Python 3.7
- Ubuntu 18.04 (The system should run on other Ubuntu versions and Windows, however not tested.)
- RBDL: Rigid Body Dynamics Library (https://rbdl.github.io/). Tested on  V.2.6.0.  (Important: set "RBDL_BUILD_ADDON_URDFREADER " to be "ON" when you compile. Also don't forget to add the compiled rbdl library in your python path use it.)
- pytorch 1.10.1
- For other python packages, please check requirements.txt

## Installation
- Download and install Python binded RBDL from  https://github.com/rbdl/rbdl
- Install Pytorch 1.8.1 with GPU support (https://pytorch.org/) (other versions should also work but not tested)
- Install python packages by:

		pip install -r requirements.txt

## How to Run 
1) Download pretrained model from [here](https://drive.google.com/file/d/1ViIDOiCkBcUUm_BIS3W1z2ELRTKYIbBQ/view?usp=sharing). Below, we assume all the pretrained networks are place under "../pretrained_neuralPhys/".
 
2) We provide a sample data under "sample_data" To run the code on our sample data, first go to root directory (neuralphyscap_demo_release) and run:

		python demo.py  --input_path sample_data/sample_dance.npy --net_path ../pretrained_neuralPhys/  --img_width 1280 --img_height 720

The predictions will be saved under "./results/"

3) To visualize the predictions, run:

		python Visualizations/simple.py 

## How to Run on Your Data
1. Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and save the prediction.


2. Process your openpose data to be compatible with NeuralPhyscap:
 
		python process_openpose.py --input_path /PATH/TO/OPENPOSE/JSON/FILE --save_path /PATH/TO/SAVE --save_file_name YOUR_DATA.npy

This will generate ".npy" input file to run NeuralPhyscap. Say we name the npy file "YOUR_DATA.npy". 

3. Run NeuralPhyscap on the generated npy file:
	 
		python demo.py  --input_path PATH/TO/YOUR_DATA.npy --net_path ../pretrained_neuralPhys/  --img_width IMAGE_WIDTH --img_height IMAGE_HEIGHT

 Replace IMAGE_WIDTH and IMAGE_HEIGHT with your own video width and height (integer values)

## License Terms
Permission is hereby granted, free of charge, to any person or company obtaining a copy of this software and associated documentation files (the "Software") from the copyright holders to use the Software for any non-commercial purpose. Publication, redistribution and (re)selling of the software, of modifications, extensions, and derivates of it, and of other software containing portions of the licensed Software, are not permitted. The Copyright holder is permitted to publically disclose and advertise the use of the software by any licensee. 

Packaging or distributing parts or whole of the provided software (including code, models and data) as is or as part of other software is prohibited. Commercial use of parts or whole of the provided software (including code, models and data) is strictly prohibited. Using the provided software for promotion of a commercial entity or product, or in any other manner which directly or indirectly results in commercial gains is strictly prohibited. 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation
If the code is used, the licesnee is required to cite the following publication in any documentation 
or publication that results from the work:
```
@article{
	PhysAwareTOG2021,
	author = {Shimada, Soshi and Golyanik, Vladislav and Xu, Weipeng and P\'{e}rez, Patrick and Theobalt, Christian},
	title = {Neural Monocular 3D Human Motion Capture with Physical Awareness},
	journal = {ACM Transactions on Graphics}, 
	month = {aug},
	volume = {40},
	number = {4}, 
	articleno = {83},
	year = {2021}, 
	publisher = {ACM}, 
	keywords = {Monocular 3D Human Motion Capture, Physical Awareness, Global 3D, Physionical Approach}
}
```