from leap.method.ab.multirate.methods import methods
from leap.method.ab.multirate.processors import MRABToTeXProcessor

for name, method in methods.items():
    #mrab2tex = MRABToTeXProcessor(method, 3, no_mixing=False)
    mrab2tex = MRABToTeXProcessor(method, 3, no_mixing=True)
    mrab2tex.run()
    open("out/%s.tex" % name, "w").write(
            "Scheme name: \\verb|%s|\n\n" % name+
            mrab2tex.get_result())

