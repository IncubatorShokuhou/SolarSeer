SolarSeer is an end-to-end AI model for 24-hour-ahead solar irradiance forecasting across the USA. The netwrok input is the past 6-hour satellite observations and the future 24-hour clear-sky solar irradiance, 
where clear-sky solar irradiance is a function of latitude, longtitude and time. The network output is the future 24-hour-ahead solar irradiance forecasts. SolarSeer can serves as a foundation model for solar PV power generation forecasting. 

To run the inference code, please follow the steps below:

1. Download the network weights and put it in the "weight" folder.
2. Download the network input, namely "satellite.npy" and "clearghi.npy". Put the two files in the "input" folder.
3. Install packages using the command ```pip install -r requirement ``` .
4. Run the 'inference.py' file. The generated forecasts will be saved in 'results' folder.
