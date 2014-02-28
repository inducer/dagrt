<TeXmacs|1.99.1>

<style|<tuple|generic|german>>

<\body>
  [Essentially a reworded expansion of p. 495 in Gear/Wells.]

  Suppose <math|y<rprime|'>=A*y>, and suppose that <math|S> diagonalizes
  <math|A>: <math|D=S*A*S<rsup|-1>>.

  Now consider a multi-stage (but single-step) method such as the midpoint
  method:

  <\equation*>
    y<rsub|n+1>=y<rsub|n>+h*f<around*|(|y<rsub|n>+<frac|h|2>f<around*|(|y<rsub|n>|)>|)>,
  </equation*>

  or for our ODE:

  <\equation*>
    y<rsub|n+1>=y<rsub|n>+h*A<around*|(|y<rsub|n>+<frac|h|2>A<around*|(|y<rsub|n>|)>|)>=<around*|(|I+h*A+<frac|<around*|(|h*A|)><rsup|2>|2>|)>y<rsub|n>=R<rsub|<with|mode|text|midpoint>><around*|(|h*A|)>y<rsub|n>,
  </equation*>

  where

  <\equation*>
    R<rsub|<text|midpoint>><around*|(|x|)>=1+x+<frac|x<rsup|2>|2>.
  </equation*>

  <math|S*R<around*|(|h*A|)>S<rsup|-1>> is obviously also diagonal, and thus
  we can consider

  <\equation*>
    <wide|y|^><rsub|n>\<assign\>S*y<rsub|n>
  </equation*>

  and find

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|y|^><rsub|n+1>>|<cell|=>|<cell|S*y<rsub|n+1>>>|<row|<cell|>|<cell|=>|<cell|S*R<rsub|<with|mode|text|midpoint>><around*|(|h*A|)>y<rsub|n>>>|<row|<cell|>|<cell|=>|<cell|S*R<rsub|<with|mode|text|midpoint>><around*|(|h*A|)>S<rsup|-1><wide|y|^><rsub|n>.>>>>
  </eqnarray*>

  So <math|<wide|y|^><rsub|n>> evolves in a component-by-component manner,
  and all stability properties can be read off from the eigenvalues of
  <math|S*R<around*|(|h*A|)>S<rsup|-1>>, which in turn are just the
  eigenvalues <math|h\<lambda\>> of <math|h*A> plugged into
  <math|R<rsub|midpoint><around*|(|\<lambda\>|)>>. The method is then stable
  iff <math|<left|\|>R<rsub|midpoint><around*|(|h\<lambda\>|)><left|\|>\<leqslant\>1>
  for each <math|\<lambda\>> and the chosen <math|h>. The region

  <\equation*>
    <left|{>z\<in\>\<bbb-C\>:<left|\|>R<rsub|midpoint><around*|(|z|)><around*|\||\<leqslant\>1|}>
  </equation*>

  is called the <with|font-shape|italic|stability region>, and <math|h> can
  be used to scale the spectrum of <math|A> such that <math|h\<lambda\>>
  falls within it. As a result, to predict whether a time integrator of this
  type is stable, it is sufficient to know

  <\itemize>
    <item>the stability region,

    <item>the eigenvalues of the ODE Jacobian <math|A>

    <item>the time step <math|h>.
  </itemize>

  This means that a large part of the stability analysis for a method
  (finding the stability region) can be done independently of the ODE being
  analyzed.

  Now consider a multi-step (but single-stage) method (such as two-step AB):

  <\equation*>
    y<rsub|n+2>=y<rsub|n+1>+<frac|3|2>h*f<around*|(|y<rsub|n+1>|)>-<frac|1|2>h*f<around*|(|y<rsub|n>|)>.
  </equation*>

  Things are a little more complicated, but not much. Let

  <\equation*>
    z<rsub|n+1>\<assign\><matrix|<tformat|<table|<row|<cell|y<rsub|n+1>>>|<row|<cell|y<rsub|n>>>>>>.
  </equation*>

  Then

  <\eqnarray*>
    <tformat|<table|<row|<cell|z<rsub|n+2>>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|y<rsub|n+2>>>|<row|<cell|y<rsub|n+1>>>>>>>>|<row|<cell|>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|y<rsub|n+1>+<frac|3|2>h*f<around*|(|y<rsub|n+1>|)>-<frac|1|2>h*f<around*|(|y<rsub|n>|)>>>|<row|<cell|y<rsub|n+1>>>>>>>>|<row|<cell|>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|y<rsub|n+1>+<frac|3|2>h*A*y<rsub|n+1>-<frac|1|2>h*A*y<rsub|n>>>|<row|<cell|y<rsub|n+1>>>>>>>>|<row|<cell|>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|<around*|(|I+<frac|3|2>h*A|)>*y<rsub|n+1>-<frac|1|2>h*A*y<rsub|n>>>|<row|<cell|y<rsub|n+1>>>>>>>>|<row|<cell|>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|I+<frac|3|2>h*A>|<cell|-<frac|1|2>h*A*>>|<row|<cell|I>|<cell|0>>>>><matrix|<tformat|<table|<row|<cell|y<rsub|n+1>>>|<row|<cell|y<rsub|n>>>>>>>>|<row|<cell|>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|I+<frac|3|2>h*A>|<cell|-<frac|1|2>h*A*>>|<row|<cell|I>|<cell|0>>>>>z<rsub|n+1>>>|<row|<cell|>|<cell|=:>|<cell|R<rsub|AB2><around*|(|h*A|)>z<rsub|n+1>>>>>
  </eqnarray*>

  Next, let

  <\equation*>
    <wide|S|^>\<assign\><matrix|<tformat|<table|<row|<cell|S>|<cell|>>|<row|<cell|>|<cell|S>>>>>
  </equation*>

  and observe that <math|<wide|S|^>R<rsub|AB2><around*|(|h*A|)><wide|S|^><rsup|-1>>
  is blockwise diagonal. Using a similar argument as above, we find that all
  stability requirement is <math|\<\|\|\>R<rsub|AB2><around*|(|h*A|)>\<\|\|\>\<leqslant\>1>,
  which, if <math|A> is diagonalizable (which we've assumed), coincides with
  <math|\<\|\|\>R<rsub|AB2><around*|(|h*\<lambda\>|)>\<\|\|\>\<leqslant\>1>,
  which again leads to the ability of finding a stability region ahead of
  time, independent of the ODE being integrated.

  The key aspect that makes this possible is that the step map/matrix is
  always a function of <math|h*A>. Multi-rate methods tend to entangle
  different step sizes, so that the step map is <with|font-shape|italic|not>
  a function of <math|h*A>. The example on the bottom end of p. 495 in Gear
  and Wells shows this.

  <with|font-series|bold|Research question:> Maybe some of our bag'o'methods
  are less entangling than others? It might be worth using some symbolic math
  (perhaps via <with|font-family|tt|sympy>) to investigate.
</body>

<\initial>
  <\collection>
    <associate|page-type|letter>
  </collection>
</initial>