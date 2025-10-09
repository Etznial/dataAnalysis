default : Out.txt Plot.svg

Out.txt data.txt: main.py
	python3 main.py > Out.txt

Plot.svg : plot.gpi data.txt 
	gnuplot $<

clean : 
	$(RM) *.dll *.exe [Oo]ut* *.svg
