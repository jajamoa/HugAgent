import json
import os
import csv
from pathlib import Path
from llm_utils import QwenLLM

def process_user_folder(user_folder, qa_start_id=1):
    """Process single user folder and generate QA data for zoning topic
    Returns: (qa_pairs, next_qa_id)
    """
    # Read demographic data
    demo_path = user_folder / "demographic" / "demographic.json"
    if not demo_path.exists():
        return [], qa_start_id
        
    with open(demo_path, 'r') as f:
        demo_data = json.load(f)
    
    # Read zoning reaction data
    zoning_path = user_folder / "survey" / "zoning_reaction.json"
    if not zoning_path.exists():
        return [], qa_start_id
        
    with open(zoning_path, 'r') as f:
        zoning_data = json.load(f)
    
    # Read transcript CSV data
    transcript_path = user_folder / "transcript" / "raw" / "zoning.csv"
    context_qas = []
    if transcript_path.exists():
        context_qas = extract_context_qas(transcript_path)
    
    # Generate QA pairs for zoning topic
    qa_pairs = []
    
    # Get prolific ID (first 6 characters)
    prolific_id = user_folder.name[:6]
    
    # Extract demographics data (remove nested prolific_id key)
    demo_info = demo_data.get(user_folder.name, demo_data)
    
    # Generate multiple belief inference questions using LLM
    belief_qas = generate_multiple_belief_inference_questions(context_qas, prolific_id)
    
    # Generate three difficulty versions for each belief inference question
    # Note: Higher difficulty = less context (harder to infer beliefs with limited information)
    difficulty_configs = [
        {"name": "simple", "context_size": len(context_qas)},  # All available context (easiest)
        {"name": "medium", "context_size": 10},  # Medium context
        {"name": "hard", "context_size": 5}  # Minimal context (hardest)
    ]
    
    for belief_qa in belief_qas:
        # Get the source QA question content to exclude it from context for this specific question only
        source_qa_question = belief_qa.get("source_qa", {}).get("question", "")
        
        for j, config in enumerate(difficulty_configs):
            # Remove only this question's source QA from context using question content match
            context_without_current_source = [qa for qa in context_qas 
                                            if qa["question"] != source_qa_question]
            
            # Select appropriate amount of context for this difficulty
            if config["context_size"] >= len(context_without_current_source):
                difficulty_context = context_without_current_source
            else:
                difficulty_context = context_without_current_source[:config["context_size"]]
            
            # Create QA entry with LLM-generated question
            qa_entry = {
                "id": f"qa_{qa_start_id:03d}",
                "prolific_id": prolific_id,
                "demographics": demo_info,
                "context_qas": difficulty_context,
                "difficulty": config["name"],
                "topic": "zoning",
                "task_type": "belief_inference",
                "task_question": belief_qa["question"],
                "answer_options": {
                    "A": "POSITIVE effect",
                    "B": "NEGATIVE effect", 
                    "C": "NO SIGNIFICANT effect"
                },
                "answer": belief_qa["answer"],
                "source_qa": belief_qa.get("source_qa", {}),
                "reasoning": belief_qa.get("reasoning", "")
            }
            qa_pairs.append(qa_entry)
            qa_start_id += 1
    return qa_pairs, qa_start_id

def extract_zoning_answer(zoning_data, user_id):
    """Extract answer from zoning reaction data"""
    # Get user's data from the nested structure
    if user_id not in zoning_data:
        return "neutral"
    
    user_data = zoning_data[user_id]
    opinions = user_data.get("opinions", {})
    
    # Calculate average opinion score (assuming 1-10 scale)
    if opinions:
        avg_score = sum(opinions.values()) / len(opinions)
        # Convert to support/oppose/neutral based on score
        if avg_score >= 7:
            return "support"
        elif avg_score <= 4:
            return "oppose"
        else:
            return "neutral"
    
    return "neutral"

def extract_context_qas(csv_path):
    """Extract QA pairs from transcript CSV file, skipping first 6 guidance questions"""
    context_qas = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = list(reader)
        
        # Find the start of QA data (after the header section)
        qa_start_idx = None
        for i, line in enumerate(lines):
            if len(line) >= 3 and line[0] == "Question Number":
                qa_start_idx = i + 1
                break
        
        if qa_start_idx is None:
            return context_qas
            
        # Extract QA pairs, starting from question 7 (skip first 6 guidance questions)
        qa_counter = 1
        for line in lines[qa_start_idx:]:
            if len(line) >= 3 and line[0].strip():  # Has question number
                question_num = int(line[0].strip())
                
                # Skip first 6 guidance questions
                if question_num > 6:
                    qa_pair = {
                        "question_number": str(qa_counter),
                        "question": line[1].strip(),
                        "answer": line[2].strip()
                    }
                    context_qas.append(qa_pair)
                    qa_counter += 1
    
    return context_qas

def generate_multiple_belief_inference_questions(context_qas, prolific_id):
    """Generate multiple belief inference questions using LLM based on context QAs"""
    if not context_qas:
        return []
    
    try:
        llm = QwenLLM(model="qwen-plus")
        
        # Format context QAs for the prompt
        context_text = "\n".join([
            f"Q{qa['question_number']}: {qa['question']}\nA{qa['question_number']}: {qa['answer']}"
            for qa in context_qas
        ])
        
        prompt = f"""
Based on the following conversation about urban zoning and housing development, identify ALL question-answer pairs that reveal the person's beliefs about causal relationships between different factors.

Conversation:
{context_text}

Your task:
1. Find ALL Q&A pairs that show how the person believes one factor affects another (up to 10 pairs)
2. For each pair, create a simple, clear question about the relationship using everyday language
3. Based on the person's answer, determine their belief about the effect

Return JSON format as an array:
[
{{
    "question": "Based on this person's responses, what do they think about the effect of [Factor A] on [Factor B]?",
    "source_qa": {{
        "question_number": "original question number",
        "question": "original question from conversation",
        "answer": "original answer from conversation"
    }},
    "answer": "A" or "B" or "C",
    "reasoning": "Brief explanation of why this answer was chosen based on their response"
}},
...
]

Where the answer options are:
- A: POSITIVE effect
- B: NEGATIVE effect  
- C: NO SIGNIFICANT effect

Use simple, everyday language for the factors. For example:
- "building more housing" instead of "upzoning policies"
- "traffic congestion" instead of "transportation systems impact"
- "neighborhood character" instead of "community identity factors"

Return up to 10 belief inference questions maximum.
"""

        response = llm.generate_response(
            prompt,
            system_message="You are an expert at analyzing conversations about urban policy to extract causal beliefs.",
            return_json=True
        )
        
        if response and isinstance(response, list):
            print(f"Generated {len(response)} belief inference questions for user {prolific_id}")
            for i, qa in enumerate(response):
                print(f"Question {i+1}: {qa.get('question', 'N/A')}")
                print(f"Answer {i+1}: {qa.get('answer', 'N/A')}")
            return response
        else:
            print(f"Failed to generate valid questions for user {prolific_id}")
            return []
            
    except Exception as e:
        print(f"Error generating belief inference questions: {e}")
        return []

def main():
    # Set up paths (relative to current working directory)
    pilot_dir = Path("./pilot_36users")
    output_dir = Path("./")
    
    # Get all user folders
    user_folders = [f for f in pilot_dir.iterdir() if f.is_dir() and not f.name.startswith('.')]
    if not user_folders:
        print("No user folders found")
        return
    
    all_qa_pairs = []
    qa_counter = 1
    
    # Loop through all user folders
    for user_folder in user_folders:
        print(f"Processing user folder: {user_folder.name}")
        
        # Process current user's data
        qa_pairs, qa_counter = process_user_folder(user_folder, qa_counter)
        all_qa_pairs.extend(qa_pairs)
        
        print(f"Generated {len(qa_pairs)} QA pairs for user {user_folder.name}")
        
        # Break after first iteration for now
        # break
    
    # Save results as JSONL (one JSON object per line, formatted)
    output_file = output_dir / "sample.jsonl"
    with open(output_file, 'w') as f:
        for qa_pair in all_qa_pairs:
            f.write(json.dumps(qa_pair, indent=2, ensure_ascii=False) + '\n')
    
    print(f"Total QA pairs generated: {len(all_qa_pairs)}")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    main()
