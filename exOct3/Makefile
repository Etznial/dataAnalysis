default : Out.txt Plot.svg

Out.txt data.data: main.py
	python3 main.py > Out.txt

Plot.svg : plot.gpi data.data 
	gnuplot $<

clean : 
	$(RM) *.dll *.exe [Oo]ut* *.data *.svg
