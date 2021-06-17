{% macro color() -%}
     \override Rest.color = #(rgb-color%{ next_color() %})
{%- endmacro %}
{% macro eps(first_bar, last_bar) -%}
_\markup {
  \general-align #Y #DOWN {
    \epsfile #X #60 #"%{ eps_waveform(first_bar, last_bar, w=5, h=0.5) %}"
  }
}
{%- endmacro %}
#(set! paper-alist (cons '("my size" . (cons (* 5.4 in) (* 1.5 in))) paper-alist))

\paper {
  #(set-paper-size "my size")
}

rs = {
  \once \override Rest.stencil = #ly:percent-repeat-item-interface::beat-slash
  \once \override Rest.thickness = #0.6
  r4
}

\header {
  tagline = ""  % removed
}


% Function to print a specified number of slashes
comp = #(define-music-function (parser location count) (integer?)
  #{
    \override Rest.stencil = #ly:percent-repeat-item-interface::beat-slash
    \override Rest.thickness = #0.48
    \override Rest.slope = #1.7
    \repeat unfold $count { r4 }
    \revert Rest.stencil
  #}
)
\score
{
<<
  \chords {a1:m e a:m e}
  \relative c'' {
\override Rest.stencil = #ly:percent-repeat-item-interface::beat-slash
\override Rest.thickness = #0.6
    %{color()%}  r4%{eps(0, 3)%}
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4
    %{color()%}  r4  \bar "|."
  }
  \addlyrics {  }
>>
\layout {
   indent = #0
   \context {
      \Score
      \override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1/16)
    }
  }
}
\version "2.18.2"  % necessary for upgrading to future LilyPond versions.
