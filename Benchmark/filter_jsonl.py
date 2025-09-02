import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from llm_utils import QwenLLM, LlamaLLM, GeminiLLM

# Minimal modules from evaluate_qwen.py
def build_belief_attribution_prompt(vqa, demographics, context_qas, include_demographics, include_context):
    """Build prompt for belief attribution task"""
    system_parts = [
        "You are an expert psychologist specializing in Theory of Mind and belief attribution.",
        "Your task: analyze conversation transcripts to infer what the participant believes about causal relationships.",
        "Focus on understanding their mental model - what they think causes what, not what is objectively true.",
        "Consider their background, conversation patterns, and implicit beliefs expressed through their responses.",
        "Base your inference strictly on evidence from their statements, not general assumptions."
    ]
    
    prompt_parts = []
    
    if include_demographics and demographics:
        demo_text = "Person's Background:\n"
        for key, value in demographics.items():
            demo_text += f"- {key.replace('_', ' ').title()}: {value}\n"
        prompt_parts.append(demo_text.strip())
    
    if include_context and context_qas:
        context_text = "Conversation History:\n"
        for i, qa in enumerate(context_qas, 1):
            context_text += f"Q{i}: {qa['question']}\n"
            context_text += f"A{i}: {qa['answer']}\n\n"
        prompt_parts.append(context_text.strip())
    
    task_question = vqa["task_question"]
    prompt_parts.append(f"Task: {task_question}")
    
    answer_options = vqa["answer_options"]
    options_text = "Answer options:\n"
    option_keys = list(answer_options.keys())
    for key in option_keys:
        options_text += f"{key}) {answer_options[key]}\n"
    prompt_parts.append(options_text.strip())
    
    options_str = ", ".join(option_keys)
    prompt_parts.append(f"Based on the evidence above, respond with ONLY the single letter ({options_str}) that best represents this person's belief.")
    
    return " ".join(system_parts), "\n\n".join(prompt_parts)

def build_belief_update_prompt(vqa, demographics, context_qas, include_demographics, include_context):
    """Build prompt for belief update task"""
    system_parts = [
        "You are an expert in survey research and human psychology.",
        "Your task: predict how this person would respond to a specific survey question based on their background and conversation.",
        "Consider their demographics, expressed opinions, and conversation patterns.",
        "Focus on understanding their likely response pattern, not what would be objectively correct.",
        "Base your prediction on evidence from their profile and statements."
    ]
    
    prompt_parts = []
    
    if include_demographics and demographics:
        demo_text = "Person's Background:\n"
        for key, value in demographics.items():
            demo_text += f"- {key.replace('_', ' ').title()}: {value}\n"
        prompt_parts.append(demo_text.strip())
    
    if include_context and context_qas:
        context_text = "Previous Conversation:\n"
        for i, qa in enumerate(context_qas, 1):
            context_text += f"Q{i}: {qa['question']}\n"
            context_text += f"A{i}: {qa['answer']}\n\n"
        prompt_parts.append(context_text.strip())
    
    task_question = vqa["task_question"]
    question_type = vqa.get("question_type", "unknown")
    
    if question_type == "opinion":
        prompt_parts.append(f"Survey Question: {task_question}")
        scale = vqa.get("scale", [1, 10])
        prompt_parts.append(f"Scale: {scale[0]} to {scale[1]}")
        prompt_parts.append(f"Based on this person's profile and conversation, what number would they likely choose? Respond with ONLY the number.")
    
    elif question_type == "reason_evaluation":
        reason_text = vqa.get("reason_text", "")
        prompt_parts.append(f"Survey Question: {task_question}")
        prompt_parts.append(f"Context: This asks about the influence of: {reason_text}")
        scale = vqa.get("scale", [1, 5])
        prompt_parts.append(f"Scale: {scale[0]} to {scale[1]} (1=no influence, {scale[1]}=very strong influence)")
        prompt_parts.append(f"Based on this person's profile and conversation, what rating would they likely give? Respond with ONLY the number.")
    
    return " ".join(system_parts), "\n\n".join(prompt_parts)

def predict_single_question(model_name, vqa, include_demographics, include_context, temperature=0):
    """Make prediction for single question"""
    try:
        # Create LLM instance
        if model_name.startswith("meta-llama"):
            llm = LlamaLLM(model=model_name)
        elif model_name.startswith("gemini"):
            llm = GeminiLLM(model=model_name)
        else:
            llm = QwenLLM(model=model_name)
        
        demographics = vqa.get("demographics", {})
        context_qas = vqa.get("context_qas", [])
        task_type = vqa.get("task_type", "belief_attribution")
        
        if task_type == "belief_attribution":
            system_message, user_prompt = build_belief_attribution_prompt(
                vqa, demographics, context_qas, include_demographics, include_context
            )
            
            generated_response = llm.generate_response(
                user_prompt, system_message=system_message, temperature=temperature
            )
            
            # Extract answer
            generated_answer = None
            if generated_response:
                response_upper = generated_response.upper()
                answer_options = vqa["answer_options"]
                option_keys = list(answer_options.keys())
                
                found_options = []
                for key in option_keys:
                    if key.upper() in response_upper:
                        found_options.append(key.upper())
                
                if len(found_options) == 1:
                    generated_answer = found_options[0]
                else:
                    for char in response_upper:
                        if char in [k.upper() for k in option_keys]:
                            generated_answer = char
                            break
            
            correct_answer = vqa['answer']
            is_correct = (generated_answer == correct_answer)
            
        elif task_type == "belief_update":
            system_message, user_prompt = build_belief_update_prompt(
                vqa, demographics, context_qas, include_demographics, include_context
            )
            
            generated_response = llm.generate_response(
                user_prompt, system_message=system_message, temperature=temperature
            )
            
            # Extract numeric answer
            generated_answer = None
            if generated_response:
                import re
                numbers = re.findall(r'\b\d+\b', generated_response.strip())
                if numbers:
                    try:
                        generated_answer = int(numbers[0])
                    except ValueError:
                        pass
            
            correct_answer = vqa['user_answer']
            
            # Evaluate correctness
            is_correct = False
            if generated_answer is not None and correct_answer is not None:
                scale = vqa.get("scale", [1, 10])
                scale_range = scale[1] - scale[0]
                tolerance = 1 if scale_range <= 5 else 2
                is_correct = abs(generated_answer - correct_answer) <= tolerance
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return {
            'generated_answer': generated_answer,
            'generated_response': generated_response,
            'correct_answer': correct_answer,
            'is_correct': is_correct
        }
        
    except Exception as e:
        return {
            'generated_answer': None,
            'generated_response': None,
            'correct_answer': vqa.get('answer') or vqa.get('user_answer', ''),
            'is_correct': False,
            'error': str(e)
        }

def process_vqa_pair(model_name, vqa, temperature=0):
    """Process single VQA with both conditions"""
    # Prediction without context and demographics
    result_no_context = predict_single_question(
        model_name, vqa, include_demographics=False, include_context=False, temperature=temperature
    )
    
    # Prediction with context and demographics
    result_with_context = predict_single_question(
        model_name, vqa, include_demographics=True, include_context=True, temperature=temperature
    )
    
    return {
        'vqa': vqa,
        'no_context': result_no_context,
        'with_context': result_with_context
    }

# Filter mechanisms
class FilterMechanisms:
    @staticmethod
    def different_results(result):
        """Keep if two predictions are different"""
        # Skip if there are errors
        if ('error' in result['no_context'] or 'error' in result['with_context']):
            return False
        no_ctx = result['no_context']['generated_answer']
        with_ctx = result['with_context']['generated_answer']
        return no_ctx != with_ctx
    
    @staticmethod
    def first_wrong_second_correct(result):
        """Keep if first prediction wrong and second correct"""
        # Skip if there are errors
        if ('error' in result['no_context'] or 'error' in result['with_context']):
            return False
        no_ctx_correct = result['no_context']['is_correct']
        with_ctx_correct = result['with_context']['is_correct']
        return (not no_ctx_correct) and with_ctx_correct
    
    @staticmethod
    def context_improves(result):
        """Keep if context/demographics improves correctness"""
        # Skip if there are errors
        if ('error' in result['no_context'] or 'error' in result['with_context']):
            return False
        no_ctx_correct = result['no_context']['is_correct']
        with_ctx_correct = result['with_context']['is_correct']
        return (not no_ctx_correct) and with_ctx_correct
    
    @staticmethod
    def context_degrades(result):
        """Keep if context/demographics makes prediction worse"""
        # Skip if there are errors
        if ('error' in result['no_context'] or 'error' in result['with_context']):
            return False
        no_ctx_correct = result['no_context']['is_correct']
        with_ctx_correct = result['with_context']['is_correct']
        return no_ctx_correct and (not with_ctx_correct)
    
    @staticmethod
    def both_wrong(result):
        """Keep if both predictions are wrong"""
        # Skip if there are errors
        if ('error' in result['no_context'] or 'error' in result['with_context']):
            return False
        no_ctx_correct = result['no_context']['is_correct']
        with_ctx_correct = result['with_context']['is_correct']
        return (not no_ctx_correct) and (not with_ctx_correct)
    
    @staticmethod
    def both_correct(result):
        """Keep if both predictions are correct"""
        # Skip if there are errors
        if ('error' in result['no_context'] or 'error' in result['with_context']):
            return False
        no_ctx_correct = result['no_context']['is_correct']
        with_ctx_correct = result['with_context']['is_correct']
        return no_ctx_correct and with_ctx_correct
    
    @staticmethod
    def has_errors(result):
        """Keep if there are processing errors"""
        return ('error' in result['no_context'] or 'error' in result['with_context'])

def analyze_accuracy_by_task_and_condition(results):
    """Analyze accuracy by task type and test conditions"""
    
    # Group results by task type
    task_stats = {}
    processed_count = 0
    error_count = 0
    
    for result in results:
        if result is None:
            continue
            
        vqa = result['vqa']
        task_type = vqa.get('task_type', 'unknown')
        
        if task_type not in task_stats:
            task_stats[task_type] = {
                'total': 0,
                'no_context_correct': 0,
                'with_context_correct': 0,
                'both_correct': 0,
                'both_wrong': 0,
                'context_improves': 0,
                'context_degrades': 0,
                'different_results': 0,
                'errors': 0
            }
        
        stats = task_stats[task_type]
        stats['total'] += 1
        processed_count += 1
        
        # Check for errors
        has_error = ('error' in result['no_context'] or 'error' in result['with_context'])
        if has_error:
            stats['errors'] += 1
            error_count += 1
            continue
        
        no_ctx_correct = result['no_context']['is_correct']
        with_ctx_correct = result['with_context']['is_correct']
        
        if no_ctx_correct:
            stats['no_context_correct'] += 1
        if with_ctx_correct:
            stats['with_context_correct'] += 1
        
        # Analyze different conditions
        if no_ctx_correct and with_ctx_correct:
            stats['both_correct'] += 1
        elif not no_ctx_correct and not with_ctx_correct:
            stats['both_wrong'] += 1
        elif not no_ctx_correct and with_ctx_correct:
            stats['context_improves'] += 1
        elif no_ctx_correct and not with_ctx_correct:
            stats['context_degrades'] += 1
        
        # Check if results are different
        no_ctx_answer = result['no_context']['generated_answer']
        with_ctx_answer = result['with_context']['generated_answer']
        if no_ctx_answer != with_ctx_answer:
            stats['different_results'] += 1
    
    print(f"Analysis summary: {processed_count} processed, {error_count} had errors")
    return task_stats

def print_accuracy_report(task_stats):
    """Print detailed accuracy report"""
    
    print(f"\n{'='*80}")
    print(f"ACCURACY ANALYSIS BY TASK TYPE AND CONDITIONS")
    print(f"{'='*80}")
    
    overall_stats = {
        'total': 0,
        'no_context_correct': 0,
        'with_context_correct': 0,
        'both_correct': 0,
        'both_wrong': 0,
        'context_improves': 0,
        'context_degrades': 0,
        'different_results': 0
    }
    
    for task_type, stats in task_stats.items():
        # Update overall stats
        for key in overall_stats:
            overall_stats[key] += stats[key]
        
        print(f"\nTask Type: {task_type.upper()}")
        print(f"{'='*50}")
        print(f"Total questions: {stats['total']}")
        
        if 'errors' in stats and stats['errors'] > 0:
            print(f"Processing errors: {stats['errors']} ({stats['errors']/stats['total']:.1%})")
        
        if stats['total'] > 0:
            # Calculate accuracy based on successfully processed questions
            valid_count = stats['total'] - stats.get('errors', 0)
            if valid_count > 0:
                no_ctx_acc = stats['no_context_correct'] / valid_count
                with_ctx_acc = stats['with_context_correct'] / valid_count
                
                print(f"No context accuracy: {no_ctx_acc:.2%} ({stats['no_context_correct']}/{valid_count})")
                print(f"With context accuracy: {with_ctx_acc:.2%} ({stats['with_context_correct']}/{valid_count})")
                print(f"Context improvement: {with_ctx_acc - no_ctx_acc:+.2%}")
                
                print(f"\nCondition breakdown (valid responses only):")
                print(f"  Both correct: {stats['both_correct']/valid_count:.1%} ({stats['both_correct']})")
                print(f"  Both wrong: {stats['both_wrong']/valid_count:.1%} ({stats['both_wrong']})")
                print(f"  Context improves: {stats['context_improves']/valid_count:.1%} ({stats['context_improves']})")
                print(f"  Context degrades: {stats['context_degrades']/valid_count:.1%} ({stats['context_degrades']})")
                print(f"  Different results: {stats['different_results']/valid_count:.1%} ({stats['different_results']})")
            else:
                print("No valid responses for analysis")
    
    # Overall summary
    if overall_stats['total'] > 0:
        print(f"\n{'='*50}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*50}")
        print(f"Total questions: {overall_stats['total']}")
        
        overall_no_ctx_acc = overall_stats['no_context_correct'] / overall_stats['total']
        overall_with_ctx_acc = overall_stats['with_context_correct'] / overall_stats['total']
        
        print(f"Overall no context accuracy: {overall_no_ctx_acc:.2%}")
        print(f"Overall with context accuracy: {overall_with_ctx_acc:.2%}")
        print(f"Overall context improvement: {overall_with_ctx_acc - overall_no_ctx_acc:+.2%}")
        
        print(f"\nOverall condition breakdown:")
        print(f"  Both correct: {overall_stats['both_correct']/overall_stats['total']:.1%}")
        print(f"  Both wrong: {overall_stats['both_wrong']/overall_stats['total']:.1%}")
        print(f"  Context improves: {overall_stats['context_improves']/overall_stats['total']:.1%}")
        print(f"  Context degrades: {overall_stats['context_degrades']/overall_stats['total']:.1%}")
        print(f"  Different results: {overall_stats['different_results']/overall_stats['total']:.1%}")

def filter_jsonl(input_path, output_path, model, filter_names, temperature=0, max_workers=3):
    """Filter JSONL based on dual predictions and filter rules"""
    
    # Load data
    vqa_dataset = []
    with open(input_path, "r") as file:
        content = file.read().strip()
        
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
    
    print(f"Loaded {len(vqa_dataset)} questions from {input_path}")
    
    # Get filter functions
    filter_funcs = []
    for filter_name in filter_names:
        if hasattr(FilterMechanisms, filter_name):
            filter_funcs.append(getattr(FilterMechanisms, filter_name))
        else:
            print(f"Warning: Unknown filter '{filter_name}', skipping")
    
    if not filter_funcs:
        print("No valid filters specified")
        return
    
    # Process all VQAs in parallel - maintain order
    print(f"Processing with model: {model}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, vqa in enumerate(vqa_dataset):
            future = executor.submit(process_vqa_pair, model, vqa, temperature)
            futures.append((i, future))
        
        # Initialize results array to maintain order and ensure all questions are included
        results = [None] * len(vqa_dataset)
        completed_count = 0
        failed_count = 0
        
        for i, future in tqdm(futures, desc="Processing"):
            try:
                result = future.result()
                results[i] = result
                completed_count += 1
            except Exception as e:
                # Create a default result for failed processing
                print(f"Warning: Failed to process question {i+1}: {e}")
                failed_count += 1
                results[i] = {
                    'vqa': vqa_dataset[i],
                    'no_context': {
                        'generated_answer': None,
                        'generated_response': None,
                        'correct_answer': vqa_dataset[i].get('answer') or vqa_dataset[i].get('user_answer', ''),
                        'is_correct': False,
                        'error': str(e)
                    },
                    'with_context': {
                        'generated_answer': None,
                        'generated_response': None,
                        'correct_answer': vqa_dataset[i].get('answer') or vqa_dataset[i].get('user_answer', ''),
                        'is_correct': False,
                        'error': str(e)
                    }
                }
    
    print(f"Processing summary: {completed_count} successful, {failed_count} failed, {len(results)} total")
    assert len(results) == len(vqa_dataset), f"Results count {len(results)} != input count {len(vqa_dataset)}"
    
    # Analyze accuracy before filtering
    print(f"\n{'='*60}")
    print(f"ANALYSIS OF ALL PREDICTIONS")
    print(f"{'='*60}")
    
    task_stats = analyze_accuracy_by_task_and_condition(results)
    print_accuracy_report(task_stats)
    
    # Apply filters
    filtered_results = []
    filtered_raw_results = []
    for result in results:
        # Check if any filter matches
        if any(filter_func(result) for filter_func in filter_funcs):
            filtered_results.append(result['vqa'])
            filtered_raw_results.append(result)
    
    print(f"\n{'='*60}")
    print(f"FILTERING RESULTS")
    print(f"{'='*60}")
    print(f"Applied filters: {', '.join(filter_names)}")
    print(f"Filtered down to {len(filtered_results)} questions ({len(filtered_results)/len(vqa_dataset)*100:.1f}%)")
    
    # Analyze accuracy of filtered results
    if filtered_raw_results:
        print(f"\n{'='*60}")
        print(f"ANALYSIS OF FILTERED PREDICTIONS")
        print(f"{'='*60}")
        
        filtered_task_stats = analyze_accuracy_by_task_and_condition(filtered_raw_results)
        print_accuracy_report(filtered_task_stats)
    
    # Save filtered results
    with open(output_path, 'w') as f:
        for vqa in filtered_results:
            f.write(json.dumps(vqa) + '\n')
    
    print(f"\nSaved filtered results to {output_path}")
    
    # Return statistics for further analysis
    return {
        'all_results': task_stats,
        'filtered_results': filtered_task_stats if filtered_raw_results else {},
        'total_questions': len(vqa_dataset),
        'filtered_questions': len(filtered_results)
    }

def analyze_only(input_path, model, temperature=0, max_workers=3):
    """Only analyze accuracy without filtering"""
    
    # Load data
    vqa_dataset = []
    with open(input_path, "r") as file:
        content = file.read().strip()
        
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
    
    print(f"Loaded {len(vqa_dataset)} questions from {input_path}")
    
    # Process all VQAs in parallel - maintain order
    print(f"Processing with model: {model}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, vqa in enumerate(vqa_dataset):
            future = executor.submit(process_vqa_pair, model, vqa, temperature)
            futures.append((i, future))
        
        # Initialize results array to maintain order and ensure all questions are included
        results = [None] * len(vqa_dataset)
        completed_count = 0
        failed_count = 0
        
        for i, future in tqdm(futures, desc="Processing"):
            try:
                result = future.result()
                results[i] = result
                completed_count += 1
            except Exception as e:
                # Create a default result for failed processing
                print(f"Warning: Failed to process question {i+1}: {e}")
                failed_count += 1
                results[i] = {
                    'vqa': vqa_dataset[i],
                    'no_context': {
                        'generated_answer': None,
                        'generated_response': None,
                        'correct_answer': vqa_dataset[i].get('answer') or vqa_dataset[i].get('user_answer', ''),
                        'is_correct': False,
                        'error': str(e)
                    },
                    'with_context': {
                        'generated_answer': None,
                        'generated_response': None,
                        'correct_answer': vqa_dataset[i].get('answer') or vqa_dataset[i].get('user_answer', ''),
                        'is_correct': False,
                        'error': str(e)
                    }
                }
    
    print(f"Processing summary: {completed_count} successful, {failed_count} failed, {len(results)} total")
    assert len(results) == len(vqa_dataset), f"Results count {len(results)} != input count {len(vqa_dataset)}"
    
    # Analyze accuracy
    task_stats = analyze_accuracy_by_task_and_condition(results)
    print_accuracy_report(task_stats)
    
    # Save detailed results to JSON for further analysis
    base_name = input_path.replace('.jsonl', '')
    clean_model_name = model.replace("/", "-")
    results_filename = f"{base_name}_analysis_{clean_model_name}.json"
    
    analysis_results = {
        'model': model,
        'total_questions': len(vqa_dataset),
        'task_stats': task_stats,
        'summary': {
            'overall_no_context_accuracy': sum(stats['no_context_correct'] for stats in task_stats.values()) / len(vqa_dataset),
            'overall_with_context_accuracy': sum(stats['with_context_correct'] for stats in task_stats.values()) / len(vqa_dataset),
            'context_improvement_rate': sum(stats['context_improves'] for stats in task_stats.values()) / len(vqa_dataset),
            'context_degradation_rate': sum(stats['context_degrades'] for stats in task_stats.values()) / len(vqa_dataset),
            'different_results_rate': sum(stats['different_results'] for stats in task_stats.values()) / len(vqa_dataset)
        }
    }
    
    with open(results_filename, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nDetailed analysis saved to {results_filename}")
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description="Filter JSONL based on dual predictions or analyze accuracy")
    parser.add_argument("input_path", help="Path to input JSONL file")
    parser.add_argument("--model", type=str, default="qwen-plus", 
                       help="Model to use for predictions")
    parser.add_argument("--temperature", type=float, default=0, 
                       help="Temperature for generation")
    parser.add_argument("--max-workers", type=int, default=3,
                       help="Number of parallel workers")
    parser.add_argument("--filters", nargs='+', 
                       choices=['different_results', 'first_wrong_second_correct', 
                               'context_improves', 'context_degrades', 'both_wrong', 'both_correct', 'has_errors'],
                       help="Filter mechanisms to apply (if not specified, only analyze)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze accuracy without filtering")
    
    args = parser.parse_args()
    
    if args.analyze_only or not args.filters:
        # Only analyze accuracy
        analyze_only(
            input_path=args.input_path,
            model=args.model,
            temperature=args.temperature,
            max_workers=args.max_workers
        )
    else:
        # Filter and analyze
        base_name = args.input_path.replace('.jsonl', '')
        filter_suffix = '_'.join(args.filters)
        output_path = f"{base_name}_filtered_{filter_suffix}.jsonl"
        
        filter_jsonl(
            input_path=args.input_path,
            output_path=output_path,
            model=args.model,
            filter_names=args.filters,
            temperature=args.temperature,
            max_workers=args.max_workers
        )

if __name__ == "__main__":
    main()
