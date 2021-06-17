#Project Imports
from ToneAssessment import *
from ScoreAssessment import score_assessment


#audio_file = 'Fly_me_files/random.wav'
#json_file = 'Fly_me_files/Fly_me_to_the_moon.json'
#lilypond_file = 'Fly_me_files/Fly_me_to_the_moon_modified.ly'

audio_file = 'Fly_me_files/21.wav'
json_file = 'json1.json'
lilypond_file = 'lilypond1.ly'

# Obtain score scores
scores_score = []
scores_score = score_assessment(audio_file, json_file, lilypond_file)

# Obtain tone scores
tone_scores = []
tone_scores = AssessTone(audio_file)

print("\n")
print("\n")
print(audio_file)
print("Tone Scores")
print("Attack Clarity:", tone_scores[1], "Dynamics Stability:", tone_scores[2],"Pitch Stability:", tone_scores[3], "Timbre Stability:", tone_scores[4],"Timbre Richness:", tone_scores[5])
print("Score Scores")
print("Overall:", scores_score["Overall"], "Rhythm:", scores_score["Rhythm"], "Tuning:", scores_score["Tuning"], "Pitch:", scores_score["Pitch"])




