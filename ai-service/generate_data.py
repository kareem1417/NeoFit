import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

n_samples = 5000

sports_list = [
    'Boxing', 'MMA', 'Football', 'Basketball',
    'Swimming', 'Tennis', 'Track & Field', 'General Fitness'
]

data = {
    'Age': np.random.randint(16, 45, n_samples),
    'Weight': np.random.randint(55, 115, n_samples),
    'Level': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], n_samples, p=[0.4, 0.4, 0.2]),
    'Goal': np.random.choice(['Weight Loss', 'Muscle Gain', 'Endurance', 'Strength', 'Agility', 'Speed'], n_samples),
    'Sport_Type': np.random.choice(sports_list, n_samples),
    'Current_Endurance_Score': np.random.randint(1, 11, n_samples),
    'Current_Strength_Score': np.random.randint(1, 11, n_samples)
}

df = pd.DataFrame(data)

def assign_program(row):
    sport = row['Sport_Type']
    goal = row['Goal']
    level = row['Level']
    endurance = row['Current_Endurance_Score']

    if sport == 'Boxing':
        if level == 'Beginner': return 'PRG_BOX_BEGINNER'
        if goal in ['Endurance', 'Weight Loss']: return 'PRG_BOX_CARDIO'
        return 'PRG_BOX_POWER'

    elif sport == 'MMA':
        if row['Current_Strength_Score'] < 5: return 'PRG_MMA_STRENGTH'
        if endurance < 5: return 'PRG_MMA_CONDITIONING'
        return 'PRG_MMA_TECHNIQUE'

    elif sport == 'Football':
        if goal == 'Speed': return 'PRG_FB_SPRINT'
        if endurance < 6: return 'PRG_FB_STAMINA'
        return 'PRG_FB_STRENGTH'

    elif sport == 'Basketball':
        if goal == 'Strength': return 'PRG_BB_POST'
        if level == 'Advanced': return 'PRG_BB_PLYO'
        return 'PRG_BB_AGILITY'

    elif sport == 'Swimming':
        if goal == 'Endurance': return 'PRG_SW_DISTANCE'
        return 'PRG_SW_SPRINT'

    elif sport == 'Tennis':
        if goal == 'Agility': return 'PRG_TN_FOOTWORK'
        return 'PRG_TN_CORE_POWER'

    elif sport == 'Track & Field':
        if goal == 'Speed' or level == 'Beginner': return 'PRG_TF_SPRINT'
        return 'PRG_TF_MARATHON'

    else:
        if goal == 'Weight Loss': return 'PRG_FIT_FAT_BURN'
        elif goal == 'Muscle Gain': return 'PRG_FIT_HYPERTROPHY'
        elif goal == 'Strength': return 'PRG_FIT_POWERLIFTING'
        else: return 'PRG_FIT_HIIT'

df['Recommended_Program_ID'] = df.apply(assign_program, axis=1)

noise_indices = df.sample(frac=0.05).index
all_programs = df['Recommended_Program_ID'].unique()
df.loc[noise_indices, 'Recommended_Program_ID'] = np.random.choice(all_programs, len(noise_indices))

df.to_csv('fitness_dataset.csv', index=False)
print(f"✅ Dataset generated successfully with {len(sports_list)} sports: fitness_dataset.csv ({n_samples} rows)")