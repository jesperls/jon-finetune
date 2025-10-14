import os
import json
import pymysql
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


class ChatDataExtractor:
    def __init__(self):
        self.connection = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
    def extract_messages(self, channel: str = None, limit: int = None) -> List[Dict]:
        with self.connection.cursor() as cursor:
            query = """
                SELECT 
                    m.id,
                    m.message,
                    m.timestamp,
                    m.channel,
                    u.username,
                    u.display_name
                FROM messages m
                JOIN users u ON m.user_id = u.id
            """
            
            conditions = []
            params = []
            
            if channel:
                conditions.append("m.channel = %s")
                params.append(channel)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY m.timestamp ASC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def create_training_examples(
        self, 
        messages: List[Dict],
        context_length: int = 10,
        min_context: int = 3
    ) -> List[Dict]:
        training_data = []
        
        print(f"Creating training examples from {len(messages)} messages...")
        
        for i in tqdm(range(min_context, len(messages))):
            context_start = max(0, i - context_length)
            context_messages = messages[context_start:i]
            target_message = messages[i]
            
            context_text = self._format_context(context_messages)
            
            #target_text = f"{target_message['username']}: {target_message['message']}"
            target_text = f"{target_message['message']}"
            
            training_data.append({
                'context': context_text,
                'response': target_text,
                'timestamp': target_message['timestamp'].isoformat() if isinstance(target_message['timestamp'], datetime) else str(target_message['timestamp']),
                'channel': target_message['channel']
            })
        
        return training_data
    
    def _format_context(self, messages: List[Dict]) -> str:
        formatted = []
        for msg in messages:
            username = msg.get('display_name') or msg.get('username')
            formatted.append(f"{username}: {msg['message']}")
        
        return "\n".join(formatted)
    
    def save_dataset(self, data: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} training examples to {output_path}")
    
    def close(self):
        self.connection.close()


def main():
    print("Starting data extraction...")
    
    extractor = ChatDataExtractor()
    
    try:
        messages = extractor.extract_messages()
        print(f"Extracted {len(messages)} messages from database")
        
        training_data = extractor.create_training_examples(
            messages,
            context_length=4,
            min_context=1
        )
        
        split_idx = int(len(training_data) * 0.9)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        os.makedirs('data', exist_ok=True)
        extractor.save_dataset(train_data, 'data/train.json')
        extractor.save_dataset(val_data, 'data/validation.json')
        
        print(f"\nDataset statistics:")
        print(f"  Training examples: {len(train_data)}")
        print(f"  Validation examples: {len(val_data)}")
        print(f"  Total: {len(training_data)}")
        
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
