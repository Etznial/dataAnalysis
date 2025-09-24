default : Out.txt Plot.svg

Out.txt decay.data decayFunc.txt decayFuncFloc.txt decayTau.txt: main.py
	python3 main.py > Out.txt

Plot.svg : plot.gpi decay.data 
	gnuplot $<

clean : 
	$(RM) *.dll *.exe [Oo]ut* *.data *.svg
