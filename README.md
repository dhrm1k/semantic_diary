# semantic notes

a simple note-taking app that actually understands what you write. instead of searching by exact words, it finds notes based on meaning.

## what it does

you write notes like "had coffee with sarah to discuss the project" and later search for "work meetings" and it'll find that note. the ai understands that coffee meetings about projects are work-related.

basically it converts your text into numbers (embeddings) that represent the meaning, then when you search it finds the closest matches.

## how to use

```bash
# set up virtual environment 
python -m venv venv
venv\Scripts\activate.bat

# install dependencies
pip install sentence-transformers faiss-cpu numpy

# run the demo
python enhanced_demo.py
```

first run takes a while because it downloads the ai model (about 30-60 seconds). after that it's fast.

## what happens when you run it

1. loads the ai model (sentencetransformer)
2. creates 20 test notes in different categories 
3. runs accuracy tests to see how well it finds related notes
4. lets you search interactively

## performance

based on testing:
- model loading: 7 seconds (one time)
- saving a note: 49ms (includes ai processing)
- searching: 33ms (very fast)
- memory usage: about 70mb for the model

## the files

- `main.py` - basic version, just stores and retrieves notes
- `demo.py` - original demo with 6 sample notes
- `enhanced_demo.py` - improved version with 20 notes and accuracy testing
- `cpu_performance_test.py` - benchmarks cpu usage during operations
- `AGENT.md` - detailed technical documentation

## how accurate is it

from the test results:
- health and fitness queries: 100% accurate
- financial topics: 100% accurate  
- people and relationships: 75% accurate
- programming topics: 33% accurate (needs better queries)
- learning topics: 33% accurate (same issue)

overall about 60% of tests pass. the ai works better with some topics than others.

## the tech stack

- **sqlite** for storing the actual notes
- **faiss** for fast similarity search on the embeddings
- **sentencetransformers** for converting text to meaning vectors
- **python** because it has good ai libraries

## why this approach

instead of keyword matching like "find notes containing 'python'", you can search for "programming stuff" and it finds notes about javascript, docker, debugging, etc. 

the downside is it takes more cpu and memory than basic text search. but for personal notes (hundreds or thousands) it's totally fine.

## gui plans

thinking about adding a tkinter interface where you:
- write notes in a text editor
- hit ctrl+s to save (generates embedding automatically)
- search box that shows results as you type
- no internet required, everything runs locally

the performance tests show it's fast enough for a smooth gui experience.

## what's next

- better test queries (the current ones are too generic)
- maybe try a larger ai model for better accuracy
- build the actual gui version
- add features like note categories, tags, dates
- possibly a web version with a proper database

## why local processing

could use openai api or similar but:
- costs money per request
- requires internet
- sends your private notes to external servers
- has latency

local processing means your notes stay private and it works offline.

the initial model download is annoying but worth it for the privacy and speed.
