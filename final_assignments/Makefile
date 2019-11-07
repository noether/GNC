all: projects.pdf

clean:
	$(RM) *~ *.pdf *.dvi *.log *.aux *.bbl *.blg *.toc *.lol *.loa *.lox \
		*.lot *.out *.lg *.tmp *.xref *.lof .*.swp

projects.pdf: projects.tex
	pdflatex projects
	bibtex projects
	pdflatex projects
	pdflatex projects
