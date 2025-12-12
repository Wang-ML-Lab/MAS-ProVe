# Library of Process Evaluation for Multi-Agent Systems (MAS Proc-Eval)


### Usage
Installing the client-server framework for MAS process evaluation:
```bash
pip install -e .
python -c "import mas_proceval"
```

Starting the Judge server: 
```bash
python -m mas_proceval.servers.server_judge
```

Using the client-server framework as a plug-in for the existing MAS systems:
```python
from mas_proceval import BaseClient, llm_parallel_search_decorator
client = BaseClient()

# this is the function we want to perform the search: 
@llm_parallel_search_decorator
def function():
    pass
```

### To-Do List
- [x] Base Client-Server Framework.
- [x] Client-Server Framework w/ Judge. 
- [x] A temporary file for managing the config: `config.yaml`.
- [ ] A simple tool for setting up the config, including the server/client ports, api keys, etc.
- [ ] (mas) Debate with our implementaiton. 
- [ ] (mas) MAS-Zero with our implementation. 