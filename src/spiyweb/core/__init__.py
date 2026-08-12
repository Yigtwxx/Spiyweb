"""Pure computation: arrays and a config in, numbers out.

Nothing in this package may touch a vector store, an embedding model, an LLM,
the filesystem or the network. That boundary is what keeps every later phase
from turning into a rewrite.
"""
