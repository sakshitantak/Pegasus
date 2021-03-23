from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

class Summarize():
    def __init__(self, model_name, torch_device):
        self.model_name = model_name
        self.torch_device = torch_device
        
    def concatenate_summarizes(self, summaries):
        final_input = ''
        for summary in summaries:
            final_input.join(summary + ' ')
        return final_input
    
    def single_document_summarization(self, src_text):
        tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(torch_device)
        batch = tokenizer(src_text,
                         truncation = True,
                          padding = True,
                          return_tensors = 'pt'
                         ).to(self.torch_device)
        
        translated = model.generate(**batch)
        generated_summary = tokenizer.batch_decode(translated,
                                                  skip_special_tokens = True
                                                  )
        return generated_summary

if __name__ == '__main__':
    model_name = 'google/pegasus-xsum'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    summarizer = Summarize(model_name, torch_device)
    src_text = [input()]
    single_document_summary = summarizer.single_document_summarization(src_text)
    print(single_document_summary)