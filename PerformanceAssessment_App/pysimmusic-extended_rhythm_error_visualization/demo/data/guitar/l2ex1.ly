{% macro color() -%}
     \override NoteHead.color = #(rgb-color%{ next_color() %})
     \override TabStaff.TabNoteHead.color =     #(rgb-color%{ current_color() %})
{%- endmacro %}

{% macro eps(first_bar, last_bar) -%}
_\markup {
  \general-align #Y #DOWN {
    \epsfile #X #56 #"%{ eps_waveform(first_bar, last_bar, w=5, h=0.5, right_border_shift=0) %}"
  }
}
{%- endmacro %}

#(set! paper-alist (cons '("my size" . (cons (* 5.27 in) (* 3 in))) paper-alist))

\paper {
  #(set-paper-size "my size")
}
\header {
  tagline = ""  % removed
}

symbols = {
  \time 4/4
  %{color()%} e,4%{eps(0, 2)%}
  %{color()%} a,
  %{color()%} d
  %{color()%} g \noBreak
  %{color()%} b \noBreak
  %{color()%} e'
  %{color()%} e'
  %{color()%} b \noBreak
  %{color()%} g
  %{color()%} d
  %{color()%} a,
  %{color()%} e,
   \bar "|."
}

\score {
  <<
    \new Staff { \clef "G_8" \symbols }
    \new TabStaff {
    \clef moderntab
    \symbols
      }
  >>
 \layout {
    \context {
      \Score
      proportionalNotationDuration = #(ly:make-moment 1/10)
    }
  }
}


\version "2.18.2"  % necessary for upgrading to future LilyPond versions.
