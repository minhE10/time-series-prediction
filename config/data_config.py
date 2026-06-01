DATASET_CONFIG = {
    "sunspots": {
        "time_col": "Date",
        "feature_cols": ["Monthly Mean Total Sunspot Number"],
        "target_cols": ["Monthly Mean Total Sunspot Number"],
        "bitcoin_resample": False
    },
    "appliances_energy": {
        "time_col": "date",
        "feature_cols": [
            "Appliances", "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3",
            "T4", "RH_4", "T5", "RH_5", "T6", "RH_6", "T7", "RH_7",
            "T8", "RH_8", "T9", "RH_9",
        ],
        "target_cols": ["Appliances", "lights"],
        "bitcoin_resample": False
    },
    "beijing_air_quality": {
        "time_col": ["year", "month", "day", "hour"],
        "feature_cols": [
            "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
            "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
        ],
        "target_cols": ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"],
        "bitcoin_resample": False
    },
    "hanoi_air_quality": {
        "time_col": "Local Time",
        "feature_cols": [
            "AQI", "CO", "NO2", "O3", "PM10", "PM25", "SO2",
            "Clouds", "Precipitation", "Pressure", "Relative Humidity",
            "Temperature", "UV Index", "Wind Speed",
        ],
        "target_cols": ["AQI", "CO", "NO2", "O3", "PM10", "PM25", "SO2"],
        "bitcoin_resample": False
    },
    "bitcoin": {
        "time_col": "Timestamp",
        "feature_cols": ["Open"],
        "target_cols": ["Open"],
        "bitcoin_resample": True
    }
}