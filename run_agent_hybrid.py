import json
import click
from rich.console import Console
from rich.progress import track
from agent.graph_hybrid import app

console = Console()

@click.command()
@click.option('--batch', required=True, help='Input JSONL file with questions')
@click.option('--out', required=True, help='Output JSONL file for results')
def main(batch: str, out: str):
    """Run the Retail Analytics Copilot in batch mode"""
    
    console.print("\n[bold cyan]🚀 Retail Analytics Copilot - Batch Mode[/bold cyan]\n")
    
    # Load questions
    questions = []
    try:
        with open(batch, 'r') as f:
            for line in f:
                questions.append(json.loads(line.strip()))
        console.print(f"✓ Loaded {len(questions)} questions from {batch}\n")
    except FileNotFoundError:
        console.print(f"[red]✗ File not found: {batch}[/red]")
        return
    except Exception as e:
        console.print(f"[red]✗ Error loading questions: {e}[/red]")
        return
    
    # Process each question
    results = []
    for i, q in enumerate(track(questions, description="Processing questions...")):
        qid = q.get('id', f'question_{i}')
        question_text = q.get('question', '')
        format_hint = q.get('format_hint', 'natural')
        
        console.print(f"\n[bold]Question {i+1}/{len(questions)}:[/bold] {qid}")
        console.print(f"[dim]{question_text}[/dim]")
        
        try:
            # Run the agent
            config = {"configurable": {"thread_id": qid}}
            state_input = {
                "id": qid,
                "question": question_text,
                "format_hint": format_hint,
                "attempts": 0,
                "node_visit_count": 0
            }
            
            result = app.invoke(state_input, config=config)
            
            # Parse answer
            if result.get("answer"):
                answer_data = json.loads(result["answer"])
                final_answer = answer_data.get("answer", "")
                citations = answer_data.get("citations", [])
                confidence = answer_data.get("confidence", 0.0)
            else:
                final_answer = "Error: No answer generated"
                citations = []
                confidence = 0.0
            
            # Format output according to contract
            output = {
                "id": qid,
                "final_answer": final_answer,
                "sql": result.get("sql_query", ""),
                "confidence": round(confidence, 2),
                "explanation": f"Classification: {result.get('classification', 'unknown')}. " +
                              (f"SQL executed successfully." if result.get("sql_result") else "RAG-based answer."),
                "citations": citations
            }
            
            results.append(output)
            console.print(f"[green]✓ Answer: {final_answer}[/green]")
            console.print(f"[dim]Confidence: {confidence:.2f} | Citations: {len(citations)}[/dim]")
            
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            output = {
                "id": qid,
                "final_answer": f"Error: {str(e)}",
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Processing failed: {str(e)}",
                "citations": []
            }
            results.append(output)
    
    # Write results
    try:
        with open(out, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        console.print(f"\n[bold green]✓ Results written to {out}[/bold green]")
        console.print(f"[dim]Processed {len(results)} questions successfully[/dim]\n")
    except Exception as e:
        console.print(f"[red]✗ Error writing results: {e}[/red]")

if __name__ == '__main__':
    main()
