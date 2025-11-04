"""Continue-1-OSS Text-to-Speech System by SVECTOR."""

__version__ = "1.0.0"

# Import and expose the main function
from .decoder import tokens_decoder_sync
from .engine_class import Continue1Model
