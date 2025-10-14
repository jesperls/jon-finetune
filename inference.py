import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from typing import List, Dict


class TwitchChatBot:
    def __init__(self, model_path: str, max_seq_length: int = 512):
        print(f"Loading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        self.model.eval()
        
        print("Model loaded successfully!")
        self.max_seq_length = max_seq_length
    
    def format_context(self, messages: List[Dict[str, str]]) -> str:
        formatted = []
        for msg in messages:
            formatted.append(f"{msg['username']}: {msg['message']}")
        return "\n".join(formatted)
    
    def generate_response(
        self,
        context_messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> str:
        context = self.format_context(context_messages)
        
        prompt = f"""{context}"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length - max_new_tokens
        ).to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response
    
    def chat_interactive(self):
        print("\n" + "="*80)
        print("Twitch Chat Bot - Interactive Mode")
        print("="*80)
        print("Commands:")
        print("  /quit - Exit the chat")
        print("  /clear - Clear conversation history")
        print("  /help - Show this help message")
        print("="*80 + "\n")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/quit":
                    print("Goodbye!")
                    break
                
                if user_input == "/clear":
                    conversation_history = []
                    print("Conversation history cleared.")
                    continue
                
                if user_input == "/help":
                    print("\nCommands:")
                    print("  /quit - Exit the chat")
                    print("  /clear - Clear conversation history")
                    print("  /help - Show this help message\n")
                    continue
                
                conversation_history.append({
                    "username": "You",
                    "message": user_input
                })
                
                context = conversation_history[-50:]
                
                response = self.generate_response(context)
                
                print(f"Bot: {response}\n")
                
                conversation_history.append({
                    "username": "Bot",
                    "message": response
                })
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Twitch chat model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./outputs/final_model",
        help="Path to the fine-tuned model"
    )
    
    args = parser.parse_args()
    
    model_path = args.model_path
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    bot = TwitchChatBot(model_path)
    bot.chat_interactive()


if __name__ == "__main__":
    main()
