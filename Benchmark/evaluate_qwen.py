import json
import time
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from llm_utils import QwenLLM

def process_single_question(llm, vqa, include_demographics, include_context, temperature):
    """Process a single question and return the result"""
    try:
        # Construct the question prompt
        task_question = vqa["task_question"]
        answer_options = vqa["answer_options"]
        demographics = vqa.get("demographics", {})
        context_qas = vqa.get("context_qas", [])
        
        # Build prompt components
        prompt_parts = []
        
        # Add system context for Theory of Mind evaluation
        prompt_parts.append("This is a Theory of Mind (ToM) benchmark evaluation. You need to infer what this person believes based on their responses and background.")
        
        # Add demographics if requested
        if include_demographics and demographics:
            demo_text = "Person's Background:\n"
            for key, value in demographics.items():
                demo_text += f"- {key.replace('_', ' ').title()}: {value}\n"
            prompt_parts.append(demo_text.strip())
        
        # Add context conversations if requested
        if include_context and context_qas:
            context_text = "Conversation History:\n"
            for i, qa in enumerate(context_qas, 1):  # Use all available context for this difficulty
                context_text += f"Q{i}: {qa['question']}\n"
                context_text += f"A{i}: {qa['answer']}\n\n"
            prompt_parts.append(context_text.strip())
        
        # Add the main task question
        prompt_parts.append(f"Task: {task_question}")
        
        # Dynamically generate answer options from the dict
        options_text = "Answer options:\n"
        option_keys = list(answer_options.keys())
        for key in option_keys:
            options_text += f"{key}) {answer_options[key]}\n"
        prompt_parts.append(options_text.strip())
        
        # Add instruction
        options_str = ", ".join(option_keys)
        prompt_parts.append(f"Please respond with either {options_str}.")
        
        # Combine all parts
        question_prompt = "\n\n".join(prompt_parts)
        
        # Generate answer using Qwen
        generated_response = llm.generate_response(
            question_prompt, 
            temperature=temperature
        )
        
        # Extract the answer dynamically from available options
        generated_answer = None
        if generated_response:
            response_upper = generated_response.upper()
            option_keys = list(answer_options.keys())
            
            # Try to find unique option key
            found_options = []
            for key in option_keys:
                if key.upper() in response_upper:
                    found_options.append(key.upper())
            
            if len(found_options) == 1:
                generated_answer = found_options[0]
            else:
                # Try to get first option key that appears
                for char in response_upper:
                    if char in [k.upper() for k in option_keys]:
                        generated_answer = char
                        break
        
        # Get the correct answer
        correct_answer = vqa['answer']
        
        # Evaluate correctness
        is_correct = (generated_answer == correct_answer)
        
        return {
            'vqa': vqa,
            'generated_answer': generated_answer,
            'generated_response': generated_response,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'context_qas_count': len(context_qas)
        }
        
    except Exception as e:
        return {
            'vqa': vqa,
            'generated_answer': None,
            'generated_response': None,
            'correct_answer': vqa.get('answer', ''),
            'is_correct': False,
            'context_qas_count': len(vqa.get('context_qas', [])),
            'error': str(e)
        }

def evaluate_belief_inference(benchmark_path, model="qwen-plus", temperature=0, include_demographics=True, include_context=True, max_workers=3):
    """
    Evaluate Theory of Mind belief inference questions using Qwen model
    """
    # Initialize Qwen LLM
    llm = QwenLLM(model=model)
    
    # Load benchmark data
    vqa_dataset = []
    with open(benchmark_path, "r") as file:
        content = file.read().strip()
        
        # Handle formatted JSON objects separated by newlines
        json_objects = []
        current_object = ""
        brace_count = 0
        
        for line in content.split('\n'):
            current_object += line + '\n'
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0 and current_object.strip():
                try:
                    json_obj = json.loads(current_object.strip())
                    json_objects.append(json_obj)
                    current_object = ""
                except json.JSONDecodeError:
                    continue
        
        vqa_dataset = json_objects
    
    # Separate data by difficulty
    difficulty_datasets = {"simple": [], "medium": [], "hard": []}
    for vqa in vqa_dataset:
        difficulty = vqa.get("difficulty", "unknown")
        if difficulty in difficulty_datasets:
            difficulty_datasets[difficulty].append(vqa)
    
    # Results for each difficulty
    all_results = {}
    
    print(f"Evaluating {len(vqa_dataset)} questions across {len(difficulty_datasets)} difficulty levels...")
    
    for difficulty, dataset in difficulty_datasets.items():
        if not dataset:
            continue
            
        print(f"\n{'='*60}")
        print(f"EVALUATING DIFFICULTY LEVEL: {difficulty.upper()}")
        print(f"{'='*60}")
        print(f"Questions in this difficulty: {len(dataset)}")
        
        correct = 0
        total = 0
        answers = []
        
        # Process questions in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all questions
            future_to_index = {
                executor.submit(process_single_question, llm, vqa, include_demographics, include_context, temperature): i
                for i, vqa in enumerate(dataset)
            }
            
            # Process completed results
            results = [None] * len(dataset)
            
            for future in tqdm(as_completed(future_to_index), total=len(dataset), desc=f"Processing {difficulty}"):
                index = future_to_index[future]
                result = future.result()
                results[index] = result
                
                # Add small delay between API calls to respect rate limits
                time.sleep(0.1)
            
            # Process results in order
            for i, result in enumerate(results):
                if result is None:
                    continue
                    
                vqa = result['vqa']
                generated_answer = result['generated_answer']
                generated_response = result['generated_response']
                correct_answer = result['correct_answer']
                is_correct = result['is_correct']
                context_qas_count = result['context_qas_count']
                
                if is_correct:
                    correct += 1
                    answers.append(1)
                else:
                    answers.append(0)
                
                total += 1
                
                # Print details for verbose mode
                print(f"\n[{difficulty.upper()}] Question {total}:")
                print(f"Context QAs: {context_qas_count}")
                print(f"Task: {vqa['task_question']}")
                print(f"Generated: {generated_answer} (Response: {generated_response})")
                print(f"Correct: {correct_answer}")
                print(f"Result: {'✓' if is_correct else '✗'}")
                print(f"Source QA: {vqa.get('source_qa', {}).get('question', 'N/A')}")
                if 'error' in result:
                    print(f"Error: {result['error']}")
                print("-" * 50)
        
        # Calculate and store results for this difficulty
        accuracy = correct / total if total > 0 else 0
        all_results[difficulty] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'answers': answers
        }
        
        print(f"\n{'='*50}")
        print(f"RESULTS FOR {difficulty.upper()} DIFFICULTY")
        print(f"{'='*50}")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Context QAs per question: {len(dataset[0]['context_qas']) if dataset else 0}")
    
    # Display summary results
    print(f"\n{'='*60}")
    print(f"SUMMARY RESULTS ACROSS ALL DIFFICULTIES")
    print(f"{'='*60}")
    
    total_all = sum(result['total'] for result in all_results.values())
    correct_all = sum(result['correct'] for result in all_results.values())
    accuracy_all = correct_all / total_all if total_all > 0 else 0
    
    print(f"Overall accuracy: {accuracy_all:.2%} ({correct_all}/{total_all})")
    print(f"\nBreakdown by difficulty:")
    for difficulty in ["simple", "medium", "hard"]:
        if difficulty in all_results:
            result = all_results[difficulty]
            context_size = len(difficulty_datasets[difficulty][0]['context_qas']) if difficulty_datasets[difficulty] else 0
            print(f"  {difficulty.capitalize():<8} ({context_size:2d} context): {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Theory of Mind belief inference benchmark with Qwen")
    parser.add_argument("--benchmark_path", type=str, default="sample.jsonl", 
                       help="Path to the benchmark JSONL file")
    parser.add_argument("--model", type=str, default="qwen-plus", 
                       choices=[
                           "qwen-max", 
                           "qwen-plus", 
                           "qwen-turbo",
                           "qwen2.5-72b-instruct",
                           "qwen2.5-32b-instruct", 
                           "qwen2.5-14b-instruct",
                           "qwen2.5-7b-instruct",
                           "qwen2.5-3b-instruct",
                           "qwen2.5-1.5b-instruct",
                           "qwen2.5-0.5b-instruct"
                       ], 
                       help="Qwen model to use")
    parser.add_argument("--temperature", type=float, default=0.1, 
                       help="Temperature for generation")
    parser.add_argument("--no-demographics", action="store_true",
                       help="Exclude demographics from prompt")
    parser.add_argument("--no-context", action="store_true",
                       help="Exclude context QAs from prompt")
    parser.add_argument("--max-workers", type=int, default=3,
                       help="Maximum number of parallel workers for API calls")
    
    args = parser.parse_args()
    
    try:
        results = evaluate_belief_inference(
            benchmark_path=args.benchmark_path,
            model=args.model,
            temperature=args.temperature,
            include_demographics=not args.no_demographics,
            include_context=not args.no_context,
            max_workers=args.max_workers
        )
        
        # Save results
        results_filename = f"evaluation_results_{args.model}_by_difficulty.json"
        with open(results_filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_filename}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
