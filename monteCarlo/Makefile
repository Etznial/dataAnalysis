default : Out.txt Plot.svg 

Out.txt : main.py
	python3 main.py > Out.txt


mc.data: main.py 
	python3 main.py --generate-data > mc.data

Plot.svg : plot.gpi mc.data
	gnuplot $<


clean : 
	$(RM) *.dll *.exe [Oo]ut* *.data *.svg
