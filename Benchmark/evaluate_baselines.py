import json
import time
import argparse
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from pathlib import Path
from llm_utils import Colors

class GlobalMajorityBaseline:
    """Global Majority baseline that always outputs the most frequent label"""
    
    def __init__(self, model_name="global-majority"):
        self.model_name = model_name
        self.label_counts = {}
        self.is_trained = False
    
    def train(self, dataset):
        """Compute label frequencies from training dataset"""
        print(f"Training {self.model_name} on {len(dataset)} examples...")
        
        # Count labels by task type
        for vqa in dataset:
            task_type = vqa.get("task_type", "belief_attribution")
            
            if task_type not in self.label_counts:
                self.label_counts[task_type] = Counter()
            
            if task_type == "belief_attribution":
                label = vqa.get("answer", "")
            elif task_type == "belief_update":
                label = vqa.get("user_answer", "")
            
            if label:
                self.label_counts[task_type][label] += 1
        
        # Find most frequent label for each task type
        self.majority_labels = {}
        for task_type, counter in self.label_counts.items():
            if counter:
                self.majority_labels[task_type] = counter.most_common(1)[0][0]
                print(f"  {task_type}: most frequent label = {self.majority_labels[task_type]} "
                      f"({counter[self.majority_labels[task_type]]}/{sum(counter.values())} = "
                      f"{counter[self.majority_labels[task_type]]/sum(counter.values()):.2%})")
        
        self.is_trained = True
    
    def generate_response(self, vqa, **kwargs):
        """Generate response using majority label"""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating responses")
        
        task_type = vqa.get("task_type", "belief_attribution")
        
        if task_type in self.majority_labels:
            return str(self.majority_labels[task_type])
        else:
            # Fallback for unseen task types
            if task_type == "belief_attribution":
                return "A"  # Default fallback
            else:
                return "5"  # Default middle value for rating scales


class RandomGuessBaseline:
    """Random Guess baseline that samples uniformly from candidate labels"""
    
    def __init__(self, model_name="random-guess", seed=42):
        self.model_name = model_name
        self.seed = seed
        self.rng = random.Random(seed)
        self.label_options = {}
        self.is_trained = False
    
    def train(self, dataset):
        """Collect all possible labels from training dataset"""
        print(f"Training {self.model_name} on {len(dataset)} examples...")
        
        # Collect unique labels by task type
        for vqa in dataset:
            task_type = vqa.get("task_type", "belief_attribution")
            
            if task_type not in self.label_options:
                self.label_options[task_type] = set()
            
            if task_type == "belief_attribution":
                # Collect answer options
                answer_options = vqa.get("answer_options", {})
                if answer_options:
                    self.label_options[task_type].update(answer_options.keys())
                
                # Also collect actual answer
                label = vqa.get("answer", "")
                if label:
                    self.label_options[task_type].add(label)
                    
            elif task_type == "belief_update":
                # For belief update, collect from scale or actual answers
                scale = vqa.get("scale", [1, 10])
                if scale and len(scale) >= 2:
                    for i in range(scale[0], scale[1] + 1):
                        self.label_options[task_type].add(i)
                
                # Also collect actual answer
                label = vqa.get("user_answer", "")
                if label is not None:
                    self.label_options[task_type].add(label)
        
        # Convert sets to sorted lists for consistent sampling
        for task_type in self.label_options:
            options = list(self.label_options[task_type])
            if task_type == "belief_update":
                # Sort numeric options
                try:
                    options = sorted([int(x) for x in options if str(x).isdigit()])
                except:
                    options = sorted(options)
            else:
                # Sort string options
                options = sorted(options)
            
            self.label_options[task_type] = options
            print(f"  {task_type}: possible labels = {self.label_options[task_type]}")
        
        self.is_trained = True
    
    def generate_response(self, vqa, **kwargs):
        """Generate response by randomly sampling from possible labels"""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating responses")
        
        task_type = vqa.get("task_type", "belief_attribution")
        
        if task_type in self.label_options and self.label_options[task_type]:
            return str(self.rng.choice(self.label_options[task_type]))
        else:
            # Fallback for unseen task types
            if task_type == "belief_attribution":
                answer_options = vqa.get("answer_options", {})
                if answer_options:
                    return str(self.rng.choice(list(answer_options.keys())))
                return "A"  # Default fallback
            else:
                scale = vqa.get("scale", [1, 10])
                if scale and len(scale) >= 2:
                    return str(self.rng.randint(scale[0], scale[1]))
                return "5"  # Default middle value


def process_single_question(model, vqa, include_demographics, include_context, temperature, debug=False, swap_data=None, max_retries=3):
    """Process a single question and return the result using baseline model"""
    try:
        # Use swapped data if swap experiment is enabled
        if swap_data:
            demographics = swap_data.get("demographics", {})
            context_qas = swap_data.get("context_qas", [])
        else:
            demographics = vqa.get("demographics", {})
            context_qas = vqa.get("context_qas", [])
        
        # Determine task type
        task_type = vqa.get("task_type", "belief_attribution")
        
        # Generate answer using baseline model
        generated_response = model.generate_response(vqa)
        
        if task_type == "belief_attribution":
            # Extract the answer from available options
            generated_answer = None
            if generated_response:
                response_upper = generated_response.upper()
                answer_options = vqa["answer_options"]
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
                    
                    # If still no match, use the response directly if it's a valid option
                    if not generated_answer and generated_response.upper() in [k.upper() for k in option_keys]:
                        generated_answer = generated_response.upper()
            
            # Get the correct answer
            correct_answer = vqa['answer']
            
            # Evaluate correctness
            is_correct = (generated_answer == correct_answer)
            
        elif task_type == "belief_update":
            # Extract numeric answer
            generated_answer = None
            if generated_response:
                try:
                    # Try to parse as integer
                    generated_answer = int(generated_response.strip())
                except ValueError:
                    # Try to extract number from response
                    import re
                    numbers = re.findall(r'\b\d+\b', generated_response.strip())
                    if numbers:
                        try:
                            generated_answer = int(numbers[0])
                        except ValueError:
                            pass
            
            # Get the correct answer
            correct_answer = vqa['user_answer']
            
            # For belief_update, we evaluate based on exact match or close proximity
            is_correct = False
            if generated_answer is not None and correct_answer is not None:
                # Exact match or within tolerance for ordinal scales
                scale = vqa.get("scale", [1, 10])
                scale_range = scale[1] - scale[0]
                tolerance = 1 if scale_range <= 5 else 2  # Stricter tolerance for smaller scales
                is_correct = abs(generated_answer - correct_answer) <= tolerance
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return {
            'vqa': vqa,
            'generated_answer': generated_answer,
            'generated_response': generated_response,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'context_qas_count': len(context_qas),
            'task_type': task_type
        }
        
    except Exception as e:
        return {
            'vqa': vqa,
            'generated_answer': None,
            'generated_response': None,
            'correct_answer': vqa.get('answer') or vqa.get('user_answer', ''),
            'is_correct': False,
            'context_qas_count': len(vqa.get('context_qas', [])),
            'task_type': vqa.get("task_type", "unknown"),
            'error': str(e)
        }


def create_swap_mapping(dataset):
    """Create mapping to swap data between consecutive prolific IDs"""
    prolific_ids = []
    id_to_data = {}
    
    for vqa in dataset:
        pid = vqa.get("prolific_id", "")
        if pid and pid not in id_to_data:
            prolific_ids.append(pid)
            id_to_data[pid] = {
                "demographics": vqa.get("demographics", {}),
                "context_qas": vqa.get("context_qas", [])
            }
    
    # Create swap mapping: each ID maps to next ID's data
    swap_mapping = {}
    for i, pid in enumerate(prolific_ids):
        next_pid = prolific_ids[(i + 1) % len(prolific_ids)]
        swap_mapping[pid] = id_to_data[next_pid]
    
    return swap_mapping


def evaluate_baseline_models(benchmark_path, model_type="global-majority", temperature=0, include_demographics=True, include_context=True, max_workers=3, debug=False, swap_experiment=False, train_ratio=0.8, seed=42):
    """
    Evaluate baseline models (Global Majority or Random Guess) on belief inference benchmark
    """
    
    # Initialize baseline model
    if model_type == "global-majority":
        model = GlobalMajorityBaseline()
    elif model_type == "random-guess":
        model = RandomGuessBaseline(seed=seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'global-majority' or 'random-guess'")
    
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
    
    # For baseline models, use the full dataset for both training and evaluation
    # This is acceptable since these are simple baselines that don't overfit
    train_dataset = vqa_dataset
    test_dataset = vqa_dataset
    
    print(f"Using full dataset: {len(vqa_dataset)} samples for both training and evaluation")
    
    # Train the baseline model on full dataset
    model.train(train_dataset)
    
    # Evaluate on full dataset (this is standard practice for simple baselines)
    vqa_dataset = test_dataset
    
    # Separate data by task type and difficulty
    task_types = set()
    difficulty_datasets = {"simple": [], "medium": [], "hard": [], "all": []}
    
    for vqa in vqa_dataset:
        task_type = vqa.get("task_type", "belief_attribution")
        task_types.add(task_type)
        
        # For belief_attribution, use context_length as difficulty
        if task_type == "belief_attribution":
            difficulty = vqa.get("context_length", "unknown")
            if difficulty not in difficulty_datasets:
                difficulty_datasets[difficulty] = []
            difficulty_datasets[difficulty].append(vqa)
        
        # For belief_update, group all together since difficulty doesn't apply the same way
        elif task_type == "belief_update":
            difficulty_datasets["all"].append(vqa)
    
    # Remove empty difficulty levels
    difficulty_datasets = {k: v for k, v in difficulty_datasets.items() if v}
    
    # Create swap mapping if swap experiment is enabled
    swap_mapping = create_swap_mapping(vqa_dataset) if swap_experiment else None
    
    # Results for each difficulty
    all_results = {}
    
    print(f"Evaluating {len(vqa_dataset)} questions across {len(difficulty_datasets)} groups...")
    print(f"Task types found: {', '.join(task_types)}")
    print(f"Using model: {model.model_name}")
    
    for difficulty, dataset in difficulty_datasets.items():
        if not dataset:
            continue
            
        print(f"\n{'='*60}")
        if difficulty == "all":
            # Get task type for this group
            sample_task_type = dataset[0].get("task_type", "unknown")
            print(f"EVALUATING TASK TYPE: {sample_task_type.upper()}")
        else:
            print(f"EVALUATING DIFFICULTY LEVEL: {difficulty.upper()}")
        print(f"{'='*60}")
        print(f"Questions in this group: {len(dataset)}")
        
        correct = 0
        total = 0
        answers = []
        absolute_errors = []  # For MAE calculation
        
        # Process questions sequentially for baseline models (they're fast)
        results = []
        for i, vqa in enumerate(dataset):
            if debug:
                print(f"\n{Colors.format('[DEBUG]', Colors.BOLD + Colors.YELLOW)} Processing question {Colors.format(str(i+1), Colors.CYAN)}/{Colors.format(str(len(dataset)), Colors.CYAN)} in {Colors.format(difficulty, Colors.GREEN)} difficulty")
            
            # Get swap data if experiment is enabled
            swap_data = None
            if swap_experiment and swap_mapping:
                pid = vqa.get("prolific_id", "")
                swap_data = swap_mapping.get(pid)
            
            result = process_single_question(model, vqa, include_demographics, include_context, temperature, debug, swap_data, max_retries=1)
            results.append(result)
        
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
            
            # Calculate absolute error for numeric tasks
            if vqa.get('task_type') == 'belief_update' and generated_answer is not None and correct_answer is not None:
                abs_error = abs(generated_answer - correct_answer)
                absolute_errors.append(abs_error)
            
            total += 1
                
            # Print details for verbose mode (only if not in debug mode)
            if not debug:
                group_label = difficulty.upper() if difficulty != "all" else vqa.get("task_type", "unknown").upper()
                print(f"\n[{group_label}] Question {total}:")
                print(f"Task Type: {vqa.get('task_type', 'unknown')}")
                print(f"Context QAs: {context_qas_count}")
                print(f"Task: {vqa['task_question']}")
                print(f"Generated: {generated_answer} (Response: {generated_response})")
                print(f"Correct: {correct_answer}")
                print(f"Result: {'✓' if is_correct else '✗'}")
                
                # Show different fields based on task type
                if vqa.get('task_type') == 'belief_attribution':
                    print(f"Source QA: {vqa.get('source_qa', {}).get('question', 'N/A')}")
                elif vqa.get('task_type') == 'belief_update':
                    print(f"Question Type: {vqa.get('question_type', 'N/A')}")
                    if 'reason_text' in vqa:
                        print(f"Reason: {vqa['reason_text']}")
                
                if 'error' in result:
                    print(f"Error: {result['error']}")
                print("-" * 50)
        
        # Calculate and store results for this difficulty
        accuracy = correct / total if total > 0 else 0
        mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
        
        all_results[difficulty] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'answers': answers,
            'mae': mae,
            'absolute_errors': absolute_errors
        }
        
        print(f"\n{'='*50}")
        if difficulty == "all":
            sample_task_type = dataset[0].get("task_type", "unknown")
            print(f"RESULTS FOR {sample_task_type.upper()} TASK")
        else:
            print(f"RESULTS FOR {difficulty.upper()} DIFFICULTY")
        print(f"{'='*50}")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        if mae is not None:
            print(f"MAE (Mean Absolute Error): {mae:.3f}")
        print(f"Context QAs per question: {len(dataset[0]['context_qas']) if dataset else 0}")
    
    # Display summary results
    print(f"\n{'='*60}")
    print(f"SUMMARY RESULTS ACROSS ALL DIFFICULTIES")
    print(f"{'='*60}")
    
    total_all = sum(result['total'] for result in all_results.values())
    correct_all = sum(result['correct'] for result in all_results.values())
    accuracy_all = correct_all / total_all if total_all > 0 else 0
    
    # Calculate overall MAE for belief_update tasks
    all_absolute_errors = []
    for result in all_results.values():
        if result['absolute_errors']:
            all_absolute_errors.extend(result['absolute_errors'])
    
    overall_mae = sum(all_absolute_errors) / len(all_absolute_errors) if all_absolute_errors else None
    
    print(f"Overall accuracy: {accuracy_all:.2%} ({correct_all}/{total_all})")
    if overall_mae is not None:
        print(f"Overall MAE: {overall_mae:.3f}")
    print(f"\nBreakdown by group:")
    for group_name in all_results:
        result = all_results[group_name]
        if group_name == "all":
            task_type = difficulty_datasets[group_name][0].get("task_type", "unknown")
            group_label = f"{task_type}"
        else:
            group_label = group_name
        
        context_size = len(difficulty_datasets[group_name][0]['context_qas']) if difficulty_datasets[group_name] else 0
        mae_str = f", MAE: {result['mae']:.3f}" if result['mae'] is not None else ""
        print(f"  {group_label.capitalize():<12} ({context_size:2d} context): {result['accuracy']:.2%} ({result['correct']}/{result['total']}){mae_str}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models on Theory of Mind belief inference benchmark")
    parser.add_argument("--benchmark_path", type=str, default="sample_prompt_v3.jsonl", 
                       help="Path to the benchmark JSONL file")
    parser.add_argument("--model", type=str, default="global-majority", 
                       choices=["global-majority", "random-guess"], 
                       help="Baseline model to use for evaluation")
    parser.add_argument("--temperature", type=float, default=0, 
                       help="Temperature parameter (not used for baselines, kept for compatibility)")
    parser.add_argument("--no-demographics", action="store_true",
                       help="Exclude demographics from prompt (not used for baselines, kept for compatibility)")
    parser.add_argument("--no-context", action="store_true",
                       help="Exclude context QAs from prompt (not used for baselines, kept for compatibility)")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="Maximum number of parallel workers (not used for baselines, kept for compatibility)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with detailed output")
    parser.add_argument("--swap-experiment", action="store_true",
                       help="Use next participant's demographics/context for prediction (extreme test)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of data to use for training baseline models")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    try:
        results = evaluate_baseline_models(
            benchmark_path=args.benchmark_path,
            model_type=args.model,
            temperature=args.temperature,
            include_demographics=not args.no_demographics,
            include_context=not args.no_context,
            max_workers=args.max_workers,
            debug=args.debug,
            swap_experiment=args.swap_experiment,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        
        # Save results with model name and dataset name
        dataset_name = Path(args.benchmark_path).stem
        results_filename = f"evaluation_results_{args.model}_{dataset_name}.json"
        with open(results_filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_filename}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
