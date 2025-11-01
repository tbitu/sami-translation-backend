"""
Translation service using TartuNLP smugri3_14 model from HuggingFace
Supports Northern Sami ↔ Norwegian translation (bidirectional)
Uses Fairseq library for TartuNLP's Finno-Ugric NMT models
"""
import torch
import logging
import os
import argparse
from fairseq.models.transformer import TransformerModel
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class TranslationService:
    """
    Manages translation models for Northern Sami ↔ Norwegian
    Uses TartuNLP smugri3_14 Finno-Ugric NMT model (bidirectional)
    """
    
    def __init__(self):
        """Initialize translation models and load to GPU if available"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        
        # Model name from HuggingFace - TartuNLP's smugri3_14 Finno-Ugric NMT model
        # This model supports bidirectional translation between 27 languages including:
        # - Northern Sami (sme_Latn)
        # - Norwegian Bokmål (nob_Latn)
        self.model_name = "tartuNLP/smugri3_14-finno-ugric-nmt"
        
        logger.info(f"Loading TartuNLP smugri3_14 model from HuggingFace: {self.model_name}")
        
        # Download model files from HuggingFace if not already cached
        model_path = snapshot_download(repo_id=self.model_name)
        logger.info(f"Model downloaded to: {model_path}")
        
        # Load the Fairseq model (bidirectional)
        # Override the fixed_dictionary path to use the downloaded model directory
        self.model = TransformerModel.from_pretrained(
            model_path,
            checkpoint_file='smugri3_14.pt',
            bpe='sentencepiece',
            sentencepiece_model='flores200_sacrebleu_tokenizer_spm.ext.model',
            data_name_or_path=model_path,
            fixed_dictionary=os.path.join(model_path, 'nllb_model_dict.ext.txt')
        )
        
        if torch.cuda.is_available():
            self.model.cuda()
        
        logger.info("✓ TartuNLP smugri3_14 translation model loaded successfully!")
    
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text between Northern Sami and Norwegian using TartuNLP smugri3_14 (bidirectional)
        
        Args:
            text: Text to translate
            src_lang: Source language code ("sme" for Northern Sami or "nor" for Norwegian)
            tgt_lang: Target language code ("sme" or "nor")
        
        Returns:
            Translated text
        """
        if src_lang == tgt_lang:
            return text
        
        # Map our language codes to TartuNLP smugri3_14 codes
        lang_map = {
            "sme": "sme_Latn",  # Northern Sami
            "nor": "nob_Latn"   # Norwegian Bokmål
        }
        
        src_code = lang_map.get(src_lang)
        tgt_code = lang_map.get(tgt_lang)
        
        if not src_code or not tgt_code:
            raise ValueError(f"Unsupported language pair: {src_lang} → {tgt_lang}")
        
        # Translate using Fairseq model (bidirectional)
        translation = self.model.translate(
            text,
            source_lang=src_code,
            target_lang=tgt_code,
            beam=5  # Beam search for better quality
        )
        
        return translation

