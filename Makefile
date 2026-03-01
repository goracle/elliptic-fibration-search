# Makefile for LaTeX documentation
# Target document
DOC = docs
SOURCE = $(DOC).tex

# Compiler and flags
LATEXMK = latexmk
FLAGS = -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make

# Default target
all: $(DOC).pdf

# Main build rule
$(DOC).pdf: $(SOURCE)
	$(LATEXMK) $(FLAGS) $(SOURCE)

# Continuous watch/preview mode (very useful for development)
# This will automatically recompile whenever you save docs.tex
watch:
	$(LATEXMK) $(FLAGS) -pvc $(SOURCE)

# Clean up auxiliary files (keep the PDF)
clean:
	$(LATEXMK) -c

# Full clean up (removes PDF as well)
distclean:
	$(LATEXMK) -C

.PHONY: all watch clean distclean
