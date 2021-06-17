import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import simmusic.extractors.guitar as guitar

def score_assessment(audio_file, json_file, lilypond_file):
    results = guitar.assess_picking_exercise(
        json_file,
        lilypond_file,
        0.0,
        audio_file,
        image_format='png')
    with open("assets/ScoreAssessmentResults.png", "wb") as out_file:
        out_file.write(results["ImageBytes"])

    print('Overall:', results['Overall'])
    print('Rhythm:', results['Rhythm'])
    print('Tuning:', results['Tuning'])
    print('Pitch:', results['Pitch'])

    return results


#score_assessment(audio_file, json_file, lilypond_file)