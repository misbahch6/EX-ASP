%#const mode=strict.

1 { assign(X,V): dom(X,V) } 1 :- var(X), not input(X,_).
assign(X,V) :- input(X,V).

&sum { X } = V :- assign(X,V).

&sum { Y } >= 0 :- relu(Y,B).
&sum { neg(Y) } >= 0 :- relu(Y,B).
&sum { W*X: elem(Y,W,X); -Y; neg(Y) } = -B :- relu(Y,B).

{ pos(Y) } :- relu(Y,B).
&sum { neg(Y) } <= 0 :-                pos(Y).
&sum {     Y  } >  0 :-                pos(Y), mode=strict.
&sum {     Y  } <= 0 :- relu(Y,_), not pos(Y).

&sum { X } >= V :- output(X,ge,V).
&sum { X } >  V :- output(X,gt,V).
&sum { X } <= V :- output(X,le,V).
&sum { X } <  V :- output(X,lt,V).

#defined input/2.
#defined output/2.

#show assign/2.
