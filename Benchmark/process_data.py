import json
import os
import csv
import argparse
import re
from pathlib import Path
from llm_utils import QwenLLM
from concurrent.futures import ThreadPoolExecutor
import threading

# Weak causal sign detector patterns
KW_TRIGGER = re.compile(
    r"(?:\b(effect|effects|impact|impacts|impacting|impacted|influence|influences|influencing|influenced|affect|affects|affecting|affected)\b"
    r"|(?:\bpositive\b|\bnegative\b)"
    r"|(?:\blead\s+to\b|\bled\s+to\b|\bcauses?\b|\bcaused\b|\bresult(?:s|ed)?\s+in\b|\bresult(?:s|ed)?\s+from\b)"
    r"|(?:\bconsequence(?:s)?\b)"
    r"|(?:\bdue\s+to\b|\bbecause\s+of\b|\bowing\s+to\b|\bas\s+a\s+result\s+of\b)"
    r"|(?:\bcausal(?:ly)?\b)"
    r"|(?:\brelationship(?:s)?\b|\bconnection(?:s)?\b))",
    re.IGNORECASE
)

def is_weak_causal_sign_question(text):
    """
    Weak identification: if keywords appear, consider it causal/effect question.
    """
    if not text or not text.strip():
        return False
    t = text.strip()
    # Keyword trigger - sufficient for detection
    return bool(KW_TRIGGER.search(t))

def process_user_folder(user_folder, qa_start_id=1, context_lengths=None, task_type="belief_attribution", topic="zoning"):
    """Process single user folder and generate QA data for specified topic and task type
    Returns: (qa_pairs, next_qa_id)
    """
    # Read demographic data
    demo_path = user_folder / "demographic" / "demographic.json"
    if not demo_path.exists():
        return [], qa_start_id
        
    with open(demo_path, 'r') as f:
        demo_data = json.load(f)
    
    # Get prolific ID (first 6 characters)
    prolific_id = user_folder.name[:6]
    
    # Extract demographics data (remove nested prolific_id key)
    demo_info = demo_data.get(user_folder.name, demo_data)
    
    qa_pairs = []
    
    if task_type == "belief_attribution":
        # Original belief attribution logic
        # Read zoning reaction data
        zoning_path = user_folder / "survey" / "zoning_reaction.json"
        if not zoning_path.exists():
            return [], qa_start_id
            
        with open(zoning_path, 'r') as f:
            zoning_data = json.load(f)
        
        # Read transcript CSV data
        transcript_path = user_folder / "transcript" / "raw" / f"{topic}.csv"
        context_qas = []
        if transcript_path.exists():
            context_qas = extract_context_qas(transcript_path)
        
        # Generate multiple belief attribution questions using LLM
        belief_qas = generate_multiple_belief_attribution_questions(context_qas, prolific_id)
        
        # Generate three context length versions for each belief attribution question
        # Note: Shorter context = harder to infer beliefs with limited information
        all_context_configs = [
            {"name": "long", "context_size": len(context_qas)},  # All available context
            {"name": "medium", "context_size": 10},  # Medium context
            {"name": "short", "context_size": 5}  # Minimal context
        ]
        
        # Filter context configs based on input parameter
        if context_lengths is None:
            context_lengths = ["short", "medium", "long"]
        
        context_configs = [config for config in all_context_configs 
                          if config["name"] in context_lengths]
        
        for belief_qa in belief_qas:
            # Get the source QA question content to exclude it from context for this specific question only
            source_qa_question = belief_qa.get("source_qa", {}).get("question", "")
            
            for j, config in enumerate(context_configs):
                # Remove only this question's source QA from context using question content match
                context_without_current_source = [qa for qa in context_qas 
                                                if qa["question"] != source_qa_question]
                
                # Select appropriate amount of context for this context length
                if config["context_size"] >= len(context_without_current_source):
                    context_length_context = context_without_current_source
                else:
                    context_length_context = context_without_current_source[:config["context_size"]]
                
                # Create QA entry with LLM-generated question
                qa_entry = {
                    "id": f"qa_{qa_start_id:03d}",
                    "prolific_id": prolific_id,
                    "demographics": demo_info,
                    "context_qas": context_length_context,
                    "context_length": config["name"],
                    "topic": topic,
                    "task_type": "belief_attribution",
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
                
    elif task_type == "belief_update":
        # New belief update logic
        # Read transcript CSV data for context
        transcript_path = user_folder / "transcript" / "raw" / f"{topic}.csv"
        context_qas = []
        if transcript_path.exists():
            context_qas = extract_context_qas(transcript_path)
        
        # Process belief update questions from survey data
        survey_qas = process_belief_update_questions(user_folder, topic)
        
        for survey_qa in survey_qas:
            qa_entry = {
                "id": f"qa_{qa_start_id:03d}",
                "prolific_id": prolific_id,
                "demographics": demo_info,
                "context_qas": context_qas,  # All context QAs without filtering
                "topic": topic,
                "task_type": "belief_update",
                "question_id": survey_qa["question_id"],
                "question_type": survey_qa["question_type"],
                "task_question": survey_qa["question_text"],
                "user_answer": survey_qa["user_answer"],
                "scale": survey_qa["scale"]
            }
            
            # Add reason-specific fields if it's a reason evaluation question
            if survey_qa["question_type"] == "reason_evaluation":
                qa_entry["reason_code"] = survey_qa["reason_code"]
                qa_entry["reason_text"] = survey_qa["reason_text"]
            
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

def load_survey_data(topic="zoning"):
    """Load survey questions and reason mappings"""
    survey_path = Path("survey_content/surveys.json")
    
    # Map topic to reason mapping file
    reason_mapping_files = {
        "zoning": "housing_reason_mapping.json",
        "surveillance": "surveillance_reason_mapping.json", 
        "healthcare": "healthcare_reason_mapping.json"
    }
    
    reason_mapping_file = reason_mapping_files.get(topic, "housing_reason_mapping.json")
    reason_mapping_path = Path(f"survey_content/{reason_mapping_file}")
    
    with open(survey_path, 'r') as f:
        survey_data = json.load(f)
    
    with open(reason_mapping_path, 'r') as f:
        reason_mapping = json.load(f)
    
    return survey_data, reason_mapping

def process_belief_update_questions(user_folder, topic="zoning"):
    """Process belief update questions based on survey data"""
    # Load survey structure and reason mappings
    survey_data, reason_mapping = load_survey_data(topic)
    
    # Map topic to survey key
    topic_map = {
        "zoning": "upzoning",
        "surveillance": "surveillance_camera", 
        "healthcare": "universal_healthcare"
    }
    
    survey_key = topic_map.get(topic, "upzoning")
    
    # Map topic to survey file name
    survey_files = {
        "zoning": "zoning_reaction.json",
        "surveillance": "camera_reaction.json",
        "healthcare": "healthcare_reaction.json"
    }
    
    # Read user's survey responses
    survey_file = survey_files.get(topic, "zoning_reaction.json")
    survey_path = user_folder / "survey" / survey_file
    
    if not survey_path.exists():
        return []
    
    with open(survey_path, 'r') as f:
        user_responses = json.load(f)
    
    user_id = user_folder.name
    if user_id not in user_responses:
        return []
    
    user_data = user_responses[user_id]
    opinions = user_data.get("opinions", {})
    reasons = user_data.get("reasons", {})
    
    # Get survey questions for this topic
    topic_questions = survey_data["topics"][survey_key]["questions"]
    
    qa_pairs = []
    
    # Process each opinion question
    for question_data in topic_questions:
        question_id = question_data["id"]
        
        if question_id in opinions:
            # Create opinion question
            opinion_qa = {
                "question_id": question_id,
                "question_type": "opinion",
                "question_text": question_data["text"],
                "user_answer": opinions[question_id],
                "scale": question_data["scale"]
            }
            qa_pairs.append(opinion_qa)
            
            # Process follow-up reason questions if they exist
            if question_data.get("has_reason_followup") and question_id in reasons:
                reason_questions = generate_reason_questions(
                    question_data, reasons[question_id], reason_mapping, question_id
                )
                qa_pairs.extend(reason_questions)
    
    return qa_pairs

def generate_reason_questions(question_data, user_reasons, reason_mapping, base_question_id):
    """Generate individual reason evaluation questions using LLM with parallel processing"""
    if "followup" not in question_data:
        return []
    
    followup = question_data["followup"]
    reason_codes = followup["reasons"]
    
    # Map reason codes to text
    mapped_reasons = []
    for code in reason_codes:
        if code in reason_mapping["reverse_mapping"]:
            reason_text = reason_mapping["reverse_mapping"][code]
            user_score = user_reasons.get(code, 0)
            mapped_reasons.append({
                "code": code,
                "text": reason_text,
                "user_score": user_score
            })
    
    if not mapped_reasons:
        return []
    
    # Use LLM to generate individual questions for each reason in parallel
    def generate_single_reason_question(reason):
        """Generate question for a single reason"""
        try:
            llm = QwenLLM(model="qwen-plus")
            
            prompt = f"""
Create a direct question that asks about how much this specific reason influences this person's opinion.

Original survey question: {question_data["text"]}
Specific reason: {reason["text"]}

Format: "How much does [specific reason] influence this person's opinion on [topic]?"

Make it:
- Direct and specific about influence level
- Start with "How much does..."
- Clearly state what influences what
- Use everyday language
- Focus on the degree/level of influence

Return only the question text, nothing else.
"""
            
            response = llm.generate_response(
                prompt,
                system_message="You are an expert at creating clear survey questions about policy opinions."
            )
            
            if response and response.strip():
                return {
                    "question_id": f"{base_question_id}r_{reason['code']}",
                    "question_type": "reason_evaluation",
                    "question_text": response.strip(),
                    "reason_code": reason["code"],
                    "reason_text": reason["text"],
                    "user_answer": reason["user_score"],
                    "scale": [1, 5]
                }
            else:
                print(f"Failed to generate question for reason {reason['code']}")
                return None
                
        except Exception as e:
            print(f"Error generating question for reason {reason['code']}: {e}")
            return None
    
    # Process all reasons in parallel
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_single_reason_question, reason) 
                      for reason in mapped_reasons]
            
            reason_questions = []
            for future in futures:
                result = future.result()
                if result is not None:
                    reason_questions.append(result)
        
        return reason_questions
        
    except Exception as e:
        print(f"Error in parallel reason question generation: {e}")
        return []

def generate_multiple_belief_attribution_questions(context_qas, prolific_id):
    """Generate multiple belief attribution questions using LLM based on context QAs"""
    if not context_qas:
        return []
    
    try:
        llm = QwenLLM(model="qwen-plus")
        
        # Filter context QAs to only include those with causal/effect patterns
        filtered_qas = [qa for qa in context_qas if is_weak_causal_sign_question(qa['question'])]
        
        if not filtered_qas:
            print(f"No causal-relevant questions found for user {prolific_id}")
            return []
        
        print(f"Filtered {len(context_qas)} QAs down to {len(filtered_qas)} causal-relevant QAs for user {prolific_id}")
        
        # Format filtered context QAs for the prompt
        context_text = "\n".join([
            f"Q{qa['question_number']}: {qa['question']}\nA{qa['question_number']}: {qa['answer']}"
            for qa in filtered_qas
        ])
        
        prompt = f"""
Based on the following conversation about urban zoning and housing development, identify ALL question-answer pairs that reveal the person's beliefs about causal relationships between different factors.

Conversation:
{context_text}

Your task:
1. Find ALL Q&A pairs that show how the person believes one factor affects another (up to 10 pairs)
2. For each pair, create a direct question asking about the influence level using everyday language
3. Based on the person's answer, determine their belief about the effect

Selection rule:
- PRIORITIZE items with dependency_level ≥ 1 (needs-context). If fewer than 10 such items exist, then fill the remainder with the best dependency_level = 0 items.
- Prefer diverse factor pairs; avoid near-duplicates.

Return JSON format as an array:
[
{{
    "question": "How much does [Factor A] affect [Factor B] according to this person?",
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
            print(f"Generated {len(response)} belief attribution questions for user {prolific_id}")
            for i, qa in enumerate(response):
                print(f"Question {i+1}: {qa.get('question', 'N/A')}")
                print(f"Answer {i+1}: {qa.get('answer', 'N/A')}")
            return response
        else:
            print(f"Failed to generate valid questions for user {prolific_id}")
            return []
            
    except Exception as e:
        print(f"Error generating belief attribution questions: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Process user data and generate QA pairs')
    parser.add_argument('--task-type', choices=['belief_attribution', 'belief_update'],
                       default='belief_attribution',
                       help='Type of task to generate (default: belief_attribution)')
    parser.add_argument('--topic', choices=['zoning', 'surveillance', 'healthcare'],
                       default='zoning',
                       help='Topic to process (default: zoning)')
    parser.add_argument('--context-lengths', nargs='+', 
                       choices=['short', 'medium', 'long'],
                       default=['short', 'medium', 'long'],
                       help='Context lengths to generate (default: all, only for belief_attribution)')
    parser.add_argument('--max-workers', type=int, default=6,
                       help='Maximum number of parallel workers (default: 6)')
    parser.add_argument('--max-users', type=int, default=10,
                       help='Maximum number of user folders to process (default: 10)')
    args = parser.parse_args()
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
    qa_counter_lock = threading.Lock()
    
    def process_user_with_counter(user_folder):
        """Thread-safe wrapper for processing user folder"""
        nonlocal qa_counter
        print(f"Processing user folder: {user_folder.name}")
        
        # Get thread-safe counter value
        with qa_counter_lock:
            current_counter = qa_counter
        
        # Process current user's data
        qa_pairs, next_counter = process_user_folder(
            user_folder, current_counter, args.context_lengths, args.task_type, args.topic
        )
        
        # Update global counter thread-safely
        with qa_counter_lock:
            qa_counter = max(qa_counter, next_counter)
        
        print(f"Generated {len(qa_pairs)} QA pairs for user {user_folder.name}")
        return qa_pairs
    
    # Process user folders concurrently with configurable workers
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for user_folder in user_folders[:args.max_users]:
            future = executor.submit(process_user_with_counter, user_folder)
            futures.append(future)
        
        # Collect results from all threads
        for future in futures:
            qa_pairs = future.result()
            all_qa_pairs.extend(qa_pairs)
    
    # Save results as JSONL (one JSON object per line, formatted)
    output_file = output_dir / f"sample_{args.task_type}_{args.topic}.jsonl"
    with open(output_file, 'w') as f:
        for qa_pair in all_qa_pairs:
            f.write(json.dumps(qa_pair, indent=2, ensure_ascii=False) + '\n')
    
    print(f"Total QA pairs generated: {len(all_qa_pairs)}")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    main()
