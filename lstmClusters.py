import enum

## This class contains all the clusters with respective features in each cluster

class ClustersFeatures(enum.Enum):
    intake_feature_names = [
                            "past_day_fats",
                            "past_day_sugars",
                            "past_day_caffeine"
                            ]
                            
    heart_feature_names = [
                            "heart_rate",
                            "ppg_std",
                            "prev_night_sleep"
                            ]

    activity_feature_names = [
                                "cumm_step_calorie",
                                "cumm_step_speed",
                                "cumm_step_distance",
                                "exercise_calorie",
                                "exercise_duration"
                                ]

    ucsd1_synthesis_feature_names = [
                                        "past_day_caffeine",
                                        "past_day_sugars",
                                        "prev_night_sleep",
                                        "ppg_std",
                                        "exercise_duration",
                                        "exercise_calorie",
                                        "cumm_step_calorie",
                                        "anxious",
                                        "distracted",
                                        "time_of_day"
                                    ]

    ucsd12_synthesis_feature_names = [
                                        "past_day_fats",
                                        "past_day_sugars",
                                        "heart_rate",
                                        "ppg_std",
                                        "cumm_step_speed",
                                        "cumm_step_calorie",
                                        "cumm_step_distance",
                                        "anxious",
                                        "distracted",
                                        "time_of_day"
                                    ]

    
    ucsd14_synthesis_feature_names = [
                                        "past_day_fats",
                                        "past_day_caffeine",
                                        "heart_rate", 
                                        "prev_night_sleep",
                                        "exercise_duration",
                                        "exercise_calorie",
                                        "anxious",
                                        "distracted",
                                        "time_of_day"
                                        ]

    
    ucsd19_synthesis_feature_names = [
                                        "past_day_fats",
                                        "past_day_sugars",
                                        "ppg_std", 
                                        "prev_night_sleep",
                                        "exercise_calorie",
                                        "cumm_step_speed",
                                        "cumm_step_calorie",
                                        "anxious",
                                        "distracted",
                                        "time_of_day"
                                        ]

    ucsd20_synthesis_feature_names = [
                                        "past_day_caffeine",
                                        "past_day_fats",
                                        "ppg_std", 
                                        "heart_rate",
                                        "exercise_duration",
                                        "cumm_step_distance",
                                        "cumm_step_speed",
                                        "anxious",
                                        "distracted",
                                        "time_of_day"
                                        ]

    ucsd26_synthesis_feature_names = [
                                        "past_day_caffeine",
                                        "past_day_fats",
                                        "prev_night_sleep", 
                                        "heart_rate",
                                        "exercise_calorie",
                                        "exercise_duration",
                                        "anxious",
                                        "distracted",
                                        "time_of_day"
                                        ]

    ucsd28_synthesis_feature_names = [
                                        "past_day_caffeine",
                                        "ppg_std", 
                                        "heart_rate",
                                        "exercise_calorie",
                                        "exercise_duration",
                                        "anxious",
                                        "distracted",
                                        "time_of_day"
                                        ]

    ucsd29_synthesis_feature_names = [
                                        "past_day_fats",
                                        "past_day_sugars",
                                        "prev_night_sleep", 
                                        "heart_rate",
                                        "exercise_duration",
                                        "exercise_calorie",
                                        "cumm_step_calorie",
                                        "anxious",
                                        "distracted",
                                        "time_of_day"
                                        ]

    all_features_names = [  
                            "anxious",
                            "distracted",
                            "past_day_fats",
                            "past_day_sugars",
                            "past_day_caffeine",
                            "heart_rate",
                            "ppg_std",
                            "prev_night_sleep",
                            "cumm_step_calorie",
                            "cumm_step_speed",
                            "cumm_step_distance",
                            "exercise_calorie",
                            "exercise_duration",
                            "time_of_day"
                            ]