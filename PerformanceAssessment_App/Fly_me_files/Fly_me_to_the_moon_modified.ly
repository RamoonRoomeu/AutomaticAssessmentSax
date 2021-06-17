\version "2.20.0"
% automatically converted by musicxml2ly from Fly_me_to_the_moon.musicxml
\pointAndClickOff

{% macro color() -%}
     \override NoteHead.color = #(rgb-color%{ next_color() %})
{%- endmacro %}
{% macro eps(scale, first_bar, last_bar) -%}
_\markup {
  \general-align #Y #DOWN {
    \epsfile #X #%{scale%} #"%{ eps_waveform(first_bar, last_bar, w=0.05*scale, h=0.35, left_border_shift=0, right_border_shift=-0.1) %}"
  }
}
{%- endmacro %}

\header {
    encodingsoftware =  "MuseScore 3.2.3"
    encodingdate =  "2021-05-26"
    composer =  "Bart Howard"
    title =  "Fly Me To The Moon"
    }

#(set-global-staff-size 20.1587428571)
\paper {
    
    paper-width = 21.0\cm
    paper-height = 29.7\cm
    top-margin = 1.0\cm
    bottom-margin = 2.0\cm
    left-margin = 1.0\cm
    right-margin = 1.0\cm
    indent = 1.61538461538\cm
    short-indent = 0.807692307692\cm
    }
\layout {
    \context { \Score
        skipBars = ##t
        autoBeaming = ##f
        proportionalNotationDuration = #(ly:make-moment 1/10)
        \override SpacingSpanner.strict-note-spacing = ##t
        \override SpacingSpanner.uniform-stretching = ##t
        }
    }
    
PartPOneVoiceOne =  \relative a'' {
    \transposition es \clef "treble" \key a \major
    \numericTimeSignature\time 4/4 | % 1
    \tempo 4=100 | % 1
    
    %{color()%}\stemDown %{eps(85, 0, 3)%} a4. %{color()%}\stemDown gis8 %{color()%}\stemDown fis4 %{color()%}\stemDown e8 [ %{color()%}\stemDown d8 ( ] | % 2
    %{color()%}\stemDown d2 ) r8 %{color()%}\stemDown e8 [ %{color()%}\stemDown fis8 %{color()%}\stemDown a8 ] | % 3
    %{color()%}\stemDown gis4. %{color()%}\stemDown fis8 %{color()%}\stemDown e4 %{color()%}\stemDown d8 [ %{color()%}\stemDown cis8 ( ] | % 4
    %{color()%}\stemDown cis2. ) r4 \break | % 5
    
    r4 %{eps(92, 4, 7)%} %{color()%}\stemDown fis8 [ %{color()%}\stemDown e8 ] %{color()%}\stemDown d8 %{color()%}\stemDown cis4. | % 6
    %{color()%}\stemDown b4 %{color()%}\stemDown cis4 %{color()%}\stemDown d8 %{color()%}\stemDown fis4. | % 7
    %{color()%}\stemDown eis4. %{color()%}\stemDown d8 %{color()%}\stemDown cis4 %{color()%}\stemUp b8 [ %{color()%}\stemUp a8 ~ ] | % 8
    %{color()%}\stemUp a2 r8 r8 %{color()%}\stemUp ais4 \break | % 9
    
    %{color()%}\stemDown b4 %{eps(92, 8, 11)%} %{color()%}\stemDown fis'8 [ %{color()%}\stemDown fis8 ~ ] %{color()%}\stemDown fis2 ~ | \barNumberCheck #10
    %{color()%}\stemDown fis2 %{color()%}\stemDown a4 %{color()%}\stemDown gis4 | % 11
    %{color()%} e1 | % 12
    r2 r4 %{color()%}\stemUp a,4 \break | % 13
    
    %{color()%}\stemDown b4 %{eps(92, 12, 15)%} %{color()%}\stemDown d8 [ %{color()%}\stemDown d8 ~ ] %{color()%}\stemDown d2 | % 14
    r8 %{color()%}\stemDown fis4. ~ %{color()%}\stemDown fis4 %{color()%}\stemDown e4 | % 15
    %{color()%}\stemDown d4. %{color()%}\stemDown cis8 ~ %{color()%}\stemDown cis2 ~ | % 16
    %{color()%}\stemDown cis2 r2 \bar "|."
    }


% The score definition
\score {
    <<
        
        \new Staff
        <<
            \set Staff.instrumentName = "Alto Sax"
            \set Staff.shortInstrumentName = "A. Sax."
            
            \context Staff << 
                \mergeDifferentlyDottedOn\mergeDifferentlyHeadedOn
                \context Voice = "PartPOneVoiceOne" {  \PartPOneVoiceOne }
                >>
            >>
        
        >>
    \layout {}
    % To create MIDI output, uncomment the following line:
    %  \midi {\tempo 4 = 100 }
    }

