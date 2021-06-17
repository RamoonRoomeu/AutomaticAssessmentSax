#Project Imports
import pandas as pd
from ToneAssessment import *
from ScoreAssessment import score_assessment

#Dash imports
import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import base64

#---------Code for dash-------------------------------------------------------------------------------------------------

#Some CSS style and
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title='How Good Did I Play'
server = app.server

dev_tools_hot_reload=False

#Define Colors
colors = {
    'background': '#FFFFFF',
    'text': '#555'
}

font = 'sans-serif'

#----------------Layout-------------------------------------------------------------------------------------------------


#app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
app.layout = html.Div(style={
    'background-image': 'url("assets/cucomber.png")',
    'background-repeat': 'no-repeat',
    'background-size': 'cover',
    'background-position': 'center',
    'verticalAlign': 'middle',
    'textAlign': 'center',
    'position': 'fixed',
    'width': '100%',
    'height': '100%',
    'top': '0px',
    'left': '0px',
    'z-index': '1000',
    'position':'absolute',

    },
    children=[
    html.H1(children='Alto Saxophone Performance Assessment',
            style={
            'textAlign': 'center',
            'color': colors['text'],
            'margin-top': '75px',
            'margin-bottom': '40px',
            'font-family': font,
            'background-size': '150'
    }),
    html.Div(children='Upload a wav file, lilypond file, and json file to start a new assessment', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    html.Div(style={
        'width':'80%',
        'margin':'auto'
    },
    children=[
        html.Div(style={
            'float':'left',
            'width':'33%',
            'margin':'auto',
        },
        children=[
            dcc.Upload(
                id='upload-data-lilypond',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files'),
                    ' (.ly file)'
                ]),
                style={
                    'width': '70%',
                    'height': 'auto',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'font-family': font,
                    'backgroundColor': '#FFFFFF',
                    'margin-top': '10px',
                    'margin-left':'auto',
                    'margin-right':'auto',
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
        ],
        ),

        html.Div(style={
            'display': 'inline-block',
            'margin':'0 auto',
            'width':'33%',
            'textAlign':'center'
        },
            children=[
            dcc.Upload(
                id='upload-data-audio',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files'),
                    ' (.wav file)'
                ]),
                style={
                    'width': '70%',
                    'height': 'auto',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'font-family': font,
                    'backgroundColor': '#FFFFFF',
                    'margin-top': '10px',
                    'margin-left': 'auto',
                    'margin-right': 'auto',
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
            ],
        ),
        html.Div(style={
            'float': 'right',
            'width': '33%',
            'margin':'auto',
        },
            children=[
                dcc.Upload(
                    id='upload-data-json',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files'),
                        ' (.json file)'
                    ]),
                    style={
                        'width': '70%',
                        'height': 'auto',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'font-family': font,
                        'backgroundColor': '#FFFFFF',
                        'margin-top': '10px',
                        'margin-left': 'auto',
                        'margin-right': 'auto',
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                ),
            ],
        ),
    ]),

    html.Div(style={
        'width': '80%',
        'margin': 'auto'
    }, children=[
        html.Div(id='loading-lilypond', style={
            'float': 'left',
            'width': '33%',
            'margin': 'auto',
            'height': '20px'
            },
        ),
        html.Div(id='loading-audio', style={
            'display': 'inline-block',
            'margin': 'auto',
            'width': '33%',
            'textAlign': 'center'
            },
        ),
        html.Div(id='loading-json', style={
            'float': 'right',
            'width': '33%',
            'margin': 'auto',
            'height': '20px'
            },
        ),
    ]),

    html.Div(id='loading', style={
        'textAlign': 'center',
        'margin-top': '20px',
    }),

#    html.Div(id='results_image', style={
#        "padding-bottom": "25px",
#    }),

    html.Div(children='Ramon Romeu Parpal - rromeu19@student.aau.dk - 2021', style={
        'textAlign': 'center',
        'color': colors['text'],
        'font-family': font,
        "position": 'absolute',
        "bottom": 0,
        'left':0,
        "height": "25px",
        "width": "100%"
    }),
])

#----------Assessment and parse functions-------------------------------------------------------------------------------

def parse_assessment(audio_contents, audio_filename, lilypond_contents, lilypond_filename, json_contents, json_filename):

    # Read and save the audiofile
    raw_audio = audio_contents.split(',')[1]
    wav_file = open("assessing_audio.wav", "wb")
    decode_audio = base64.b64decode(raw_audio)
    wav_file.write(decode_audio)
    # Read and save the lilypond
    raw_lily = lilypond_contents.split(',')[1]
    lily_file = open("lilypond2.ly", "wb")
    decode_lily = base64.b64decode(raw_lily)
    lily_file.write(decode_lily)
    # Read and save the json
    raw_json = json_contents.split(',')[1]
    json_file = open("json2.json", "wb")
    decode_json = base64.b64decode(raw_json)
    json_file.write(decode_json)

    # Obtain tone scores
    scores = []
    scores = AssessTone('assessing_audio.wav')
    #scores = [1, 1, 1, 1, 1, 1]
    goodsoundscore = scores[0]
    tonescores = scores[1:]
    categories = ['Attack Clarity', 'Dynamics <br> Stability', 'Pitch <br> Stability', 'Timbre <br> Stability', 'Timbre <br> Richness']
    df = pd.DataFrame(dict(r=tonescores, theta=categories))
    # Create figure
    fig = go.Figure(px.line_polar(df, r='r', theta='theta', line_close=True, title='Tone Assessment Results'))
    fig.update_traces(fill='toself')
    fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), showlegend=True,
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      title_x=0.5,
                      font=dict(
                          family=font,
                          size=22,
                          color=colors['text']
                      )
                      ),
    fig.write_image("ToneAssessmentResults.png")
    encoded_image_TA = base64.b64encode(open("ToneAssessmentResults.png", 'rb').read())

    # Obtain score scores
    audio_file = 'assessing_audio.wav'
    json_file = 'json2.json'
    lilypond_file = 'lilypond1.ly'
    scores_score = []
    scores_score = score_assessment(audio_file, json_file, lilypond_file)
    encoded_image_SA = base64.b64encode(open("ScoreAssessmentResults.png", 'rb').read())

    return scores_score

#--------------Callbacks-------------------------------------------------------------------------------------------------

@app.callback(Output('loading-audio', 'children'),
              Input('upload-data-audio', 'contents'),
              State('upload-data-audio', 'filename'))

def loading_audio(audio_contents, audio_filename):
    if audio_contents is None and audio_filename is None:
        raise dash.exceptions.PreventUpdate
    else:
        if os.path.splitext(audio_filename)[1] != '.wav':
            children = [html.Div([
                html.H6('Please upload a .wav file',
                        style={
                            'font-size': '13px',
                            'color': '#ca351d',
                            'font-family': font,
                        })
            ])]
            return children
        else:
            # Read and save the audiofile
            rawdata = audio_contents.split(',')[1]
            print(audio_contents[0:100])
            wav_file = open("assets/assessing_audio.wav", "wb")
            decode_string = base64.b64decode(rawdata)
            wav_file.write(decode_string)
            print("wav conversion done")

            children = [html.Div([
                html.H6(u'Uploaded file: "{}"'.format(audio_filename),
                        style={
                            'color': colors['text'],
                            'font-family': font,
                            'font-size': '16px',
                        }),
                html.Audio(
                    controls=True,
                    src= "assets/assessing_audio.wav",
                    title = audio_filename
                ),
            ])]
            return children


@app.callback(Output('loading-lilypond', 'children'),
              Input('upload-data-lilypond', 'contents'),
              State('upload-data-lilypond', 'filename'))

def loading_lily(lilypond_contents, lilypond_filename):
    if lilypond_contents is None and lilypond_filename is None:
        raise dash.exceptions.PreventUpdate
    else:
        if os.path.splitext(lilypond_filename)[1] != '.ly':
            children = [html.Div([
                html.H6('Please upload a .ly file',
                        style={
                            'font-size': '13px',
                            'color': '#ca351d',
                            'font-family': font,
                        })
            ])]
            return children
        else:
            children = [html.Div([
                html.H6(
                    u'Uploaded file: "{}"'.format(lilypond_filename),
                        style={
                            'color': colors['text'],
                            'font-family': font,
                            'font-size': '16px',
                        }),
            ])]
            return children



@app.callback(Output('loading-json', 'children'),
              Input('upload-data-json', 'contents'),
              State('upload-data-json', 'filename'))

def loading_json(json_contents, json_filename):
    if json_contents is None and json_filename is None:
        raise dash.exceptions.PreventUpdate
    else:
        if os.path.splitext(json_filename)[1] != '.json':
            children = [html.Div([
                html.H6('Please upload a .json file',
                        style={
                            'font-size': '13px',
                            'color': '#ca351d',
                            'font-family': font,
                        })
            ])]
            return children
        else:
            children = [html.Div([
                html.H6(
                    u'Uploaded file: "{}"'.format(json_filename),
                        style={
                            'color': colors['text'],
                            'font-family': font,
                            'font-size': '16px',
                        }),
            ])]
            return children

@app.callback(Output('loading', 'children'),
              Input('upload-data-audio', 'contents'),
              Input('upload-data-lilypond', 'contents'),
              Input('upload-data-json', 'contents'),
              State('upload-data-audio', 'filename'),
              State('upload-data-lilypond', 'filename'),
              State('upload-data-json', 'filename'))

def update_graph(audio_contents, lilypond_contents, json_contents, audio_filename, lilypond_filename, json_filename):
    if audio_contents is None or audio_filename is None \
            or lilypond_contents is None or lilypond_filename is None \
            or json_contents is None or json_filename is None:
        raise dash.exceptions.PreventUpdate
    else:
        if os.path.splitext(audio_filename)[1] == '.wav' \
                and os.path.splitext(lilypond_filename)[1] == '.ly' \
                and os.path.splitext(json_filename)[1] == '.json':
            print('oli')
            children = [html.Div([
                html.H6(
                    u'Now assessing "{}". This process should take a few seconds. '.format(audio_filename),
                    style={
                        'color': colors['text'],
                        'font-family': font,
                        'font-size': '16px',
                        'margin-top:':'100px'
                    }),
            ])]
            return children


@app.callback(Output('results_image', 'children'),
              Input('upload-data-audio', 'contents'),
              Input('upload-data-lilypond', 'contents'),
              Input('upload-data-json', 'contents'),
              State('upload-data-audio', 'filename'),
              State('upload-data-lilypond', 'filename'),
              State('upload-data-json', 'filename'))

def update_graph(audio_contents, lilypond_contents, json_contents, audio_filename, lilypond_filename, json_filename):
    if audio_contents is None or audio_filename is None \
            or lilypond_contents is None or lilypond_filename is None \
            or json_contents is None or json_filename is None:
        raise dash.exceptions.PreventUpdate
    else:
        if os.path.splitext(audio_filename)[1] == '.wav' \
                and os.path.splitext(lilypond_filename)[1] == '.ly' \
                and os.path.splitext(json_filename)[1] == '.json':
            # Perform and return Performance Assessment

            scores_score = parse_assessment(audio_contents, audio_filename, lilypond_contents, lilypond_filename, json_contents, json_filename)
            encoded_image_TA = base64.b64encode(open("ToneAssessmentResults.png", 'rb').read())

            children = [html.Div([
                html.H6(u'Assessment finished. Here are the results for the submitted file ({}):'.format(audio_filename),
                style={
                    'textAlign': 'center',
                    'color': colors['text'],
                    'margin-top': '25px',
                    'font-size': '16px',
                    'font-family': font,
                }),
                #Start division
                html.Div(children=[
                    #Left column of second row
                    html.Div(children =[
                        # HTML images accept base64 encoded strings in the same format
                        # that is supplied by the upload
                        html.H6(f'Overall Score: {scores_score["Overall"]}',
                        style = {
                            'textAlign': 'right',
                            'color': colors['text'],
                            'margin-top': '10px',
                            'font-size': '25px',
                            'font-family': font,
                        }),
                        html.H6(f'Rhythm: {scores_score["Rhythm"]} ',
                        style = {
                            'textAlign': 'right',
                            'color': colors['text'],
                            'margin-top': '10px',
                            'font-size': '25px',
                            'font-family': font,
                        }),
                        html.H6(f'Pitch: {scores_score["Pitch"]}',
                        style = {
                            'textAlign': 'right',
                            'color': colors['text'],
                            'margin-top': '10px',
                            'font-size': '25px',
                            'font-family': font,
                        }),
                        html.H6(f' Tuning: {scores_score["Tuning"]}',
                            style={
                                'textAlign': 'right',
                                'color': colors['text'],
                                'margin-top': '10px',
                                'font-size': '25px',
                                'font-family': font,
                            }),
                        ],
                        style={
                            'display': 'inline-block',
                            #'float': 'left',
                            'width': '49%',
                            'margin': 'auto',
                            'vertical-align': 'middle',
                            #'border': '2px black solid',
                            },
                        ),
                    #Right Column of second row
                    html.Div(children =[
                        html.Img(src='data:image/png;base64,{}'.format(encoded_image_TA.decode()),
                                 style={
                                     'margin': 'auto',
                                     'display': 'block',
                                     'background': 'transparent',
                                     'width': '99%',
                                     'height': 'auto',
                                 }),
                    ],
                        style={
                            'display': 'inline-block',
                            #'float': 'right',
                            'width': '49%',
                            'margin': 'auto',
                            'vertical-align': 'middle',
                            #'border': '2px black solid',
                            },
                    ),
                ],
                    style={
                        'width': '50%',
                        'margin': 'auto',
                        #'border': '2px black solid',
                    },
                ),

                #Third row
                html.Div(children=[
                    html.H6(u'Click on the following link to see a more detailed assessment:',
                            style={
                                'textAlign': 'center',
                                'color': colors['text'],
                                'margin-top': '25px',
                                'font-size': '16px',
                                'font-family': font,
                            }),
                    html.A('Score Assessment', download='ScoreAssessment', href='assets/ScoreAssessmentResults.png'),
                ]),

            ])]

            return children

if __name__ == '__main__':
    app.run_server(debug=False)
