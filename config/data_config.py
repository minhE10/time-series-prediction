DATASET_CONFIG = {
    "sunspots": {
        "time_col": "Date",
        "drop_cols": ["Unnamed: 0"],
        "feature_cols": [
            "Monthly Mean Total Sunspot Number",
            "month_sin", "month_cos"
        ],
        "target_cols": ["Monthly Mean Total Sunspot Number"]
    },

    "appliances_energy": {
        "time_col": "date",
        "feature_cols": [
            "Appliances", "lights",
            "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T_out",
            "RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9", "RH_out",
            "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint",
            "hour_sin", "hour_cos"
        ],
        "target_cols": ["Appliances", "lights"]
    },

    "beijing_air_quality": {
        "time_col": ["year", "month", "day", "hour"],
        "feature_cols": [
            "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
            "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
            "hour_sin", "hour_cos",
            "month_sin", "month_cos"
        ],
        "target_cols": ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    },

    "hanoi_air_quality": {
        "time_col": "Local Time",
        "feature_cols": [
            "PM25", "PM10", "AQI", "CO", "NO2", "O3", "SO2",
            "Clouds", "Precipitation", "Pressure",
            "Relative Humidity", "Temperature", "UV Index", "Wind Speed",
            "hour_sin", "hour_cos",
            "month_sin", "month_cos"
        ],
        "target_cols": ["PM25", "PM10", "AQI", "CO", "NO2", "O3", "SO2"]
    },

    "bitcoin": {
        "time_col": "Timestamp",  
        "feature_cols": ["Open"],  
        "target_cols": ["Open"]
    }
}