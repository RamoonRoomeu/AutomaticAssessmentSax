\version "2.20.0"
% automatically converted by musicxml2ly from C_Major_Scale.mxl
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
    encodingdate =  "2020-09-23"
    title =  "C Major Scale"
    }

#(set-global-staff-size 20.1587428571)
\paper {
    
    paper-width = 21.01\cm
    paper-height = 29.69\cm
    top-margin = 1.0\cm
    bottom-margin = 2.0\cm
    left-margin = 1.0\cm
    right-margin = 1.0\cm
    indent = 1.61615384615\cm
    short-indent = 1.29292307692\cm
    }
\layout {
    \context { \Score
                 proportionalNotationDuration = #(ly:make-moment 1/10)
         \override SpacingSpanner.strict-note-spacing = ##t
         \override SpacingSpanner.uniform-stretching = ##t
        }
    }
PartPOneVoiceOne =  \relative c' {
    \clef "treble" \key c \major \numericTimeSignature\time 4/4 | % 1
    \tempo 4=68 | % 1
    %{color()%}\stemUp c4 %{eps(89, 0, 3)%} \stemUp d4 %{color()%}\stemUp e4 %{color()%}\stemUp f4 | % 2
    %{color()%}\stemUp g4 %{color()%}\stemUp a4 %{color()%}\stemDown b4 %{color()%}\stemDown c4 | % 3
    %{color()%}\stemDown b4 %{color()%}\stemUp a4 %{color()%}\stemUp g4 %{color()%}\stemUp f4 | % 4
    %{color()%}\stemUp e4 %{color()%}\stemUp d4 %{color()%}\stemUp c2 \bar "|."
    }


% The score definition
\score {
    <<
        
        \new Staff
        <<
            \set Staff.instrumentName = "Sax"
            \set Staff.shortInstrumentName = "Sax."
            
            \context Staff << 
                \mergeDifferentlyDottedOn\mergeDifferentlyHeadedOn
                \context Voice = "PartPOneVoiceOne" {  \PartPOneVoiceOne }
                >>
            >>
        
        >>
    \layout {}
    % To create MIDI output, uncomment the following line:
    %  \midi {\tempo 4 = 67 }
    }

