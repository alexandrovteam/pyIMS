# Export to imzML - README

This folder contains a collection of scripts and tools for exporting data from various sources into imzML. In general the usage pattern is
``` shell
python pyImagingMSpec/convert/<input_format> <input_filename> <*parameter_file*> <output_filename>
```

If all the required information is stored in a single vendor file then this is all that is required, otherwise a parameter file that collects this information together is needed.
The following sections detail the requirements for each file type

## General install
### Requirements
* python 3.6 (easiest way to install is through [anaconda](https://anaconda.org/))
* MSConvert (for exporting files to mzML)
* [Download](https://github.com/alexandrovteam/pyImagingMSpec/archive/master.zip) the latest development version of this python library (pyImagingMSpec) 
    * unzip it
    * open cmd.exe (windows) or terminal (mac/linux)
    ``` shell 
        cd ~/Downloads/pyImagingMSpec-master (or wherever you unzipped to)  
        pip install -e . 
    ```
    * this *should* install all other python dependencies 

 
## Links to specific file type
[nanoDESI (.raw)](#nanodesi)


## nanodesi (.raw)
Data files are acquired as Thermo .raw and have been exported to mzML.
The exporter converts acquisition time into pysical locations and tries to fit the 
 best grid possible. This process assumes that the image acquisition shape is rectangular and 
  contains roughly the same number of pixels per row. 
#### Exporting steps:
1. Export all .raw files to .mzML using MSConvert
    * export peak picked
    * all files should start with the same name and be numbered by acquisition row e.g:
        * my_dataset_1.mzML
        * my_dataset_2.mzML
        * my_dataset_3.mzML
2. Create the config file (see below for an example)
3. from the command line:
   ``` python pyImagingMSpec/convert/nanoDESI <config_filename> <imzml_filename>```   
   (replacing <...> with the full file path)
   
Example config file  
*my_dataset.json*
```json
  {
  "data_folder": "~/Documents/data/nanodesi",
  "data_name": "my_dataset",
  "stage_parameters": {
      "x_velocity": "40",
      "x_velocity_units": "micrometers per second",
      "y_spacing": "100",
      "y_spacing_units": "micrometers"
      }
  }
```
If data_folder or data_name are missing from the json file, 
the exporter will assume the datasets are in the same location and start with the same name.

If the pixel dimensions in *x* and *y* are not equal then images generated from the imzML coordinates
 will need to have the correct aspect ratio set during visualisation. 
The raw data coordinates are stored in micrometers under *userParam* elements for each spectrum  
``` 
<userParam name="xCoord" value=""/>
<userParam name="yCoord" value=""/>
```