# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

from common_imports import *
import operator 
from config import PROJECT_CONFIG, MODELS_DIR, RAW_DIR, LOG_DIR

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'agents.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Global primitive set for GP
def create_pset() -> gp.PrimitiveSetTyped:
    """Create a strongly-typed primitive set for GP."""
    pset = gp.PrimitiveSetTyped("MAIN", [float], tuple)
    pset.addPrimitive(operator.add, [float, float], float, name="add")
    pset.addPrimitive(operator.sub, [float, float], float, name="sub")
    pset.addPrimitive(operator.mul, [float, float], float, name="mul")
    pset.addPrimitive(operator.truediv, [float, float], float, name="div")
    pset.addPrimitive(np.sin, [float], float, name="sin")
    pset.addPrimitive(np.cos, [float], float, name="cos")
    pset.addPrimitive(lambda x: x ** 2, [float], float, name="square")
    pset.addPrimitive(lambda a, b: (a, b), [float, float], tuple, name="output")
    pset.addTerminal(0.0, float)
    pset.addTerminal(1.0, float)
    pset.renameArguments(ARG0='x')
    return pset

# Custom mutation functions
def mutate_subtree(individual: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped) -> Tuple[gp.PrimitiveTree]:
    """Mutate one of the two subtrees under the 'output' root node."""
    if len(individual) < 3:
        logger.warning("Individual too small to mutate safely.")
        return (individual,)
    subtree_idx = random.choice([0, 1])
    slice_ = individual.searchSubtree(1) if subtree_idx == 0 else individual.searchSubtree(individual.searchSubtree(1).stop)
    point = random.randint(slice_.start, slice_.stop - 1)
    sub_slice = individual.searchSubtree(point)
    expr = gp.genGrow(pset, min_=1, max_=4, type_=float)
    new_subtree = gp.PrimitiveTree(expr)
    individual[sub_slice] = new_subtree
    logger.info(f"Mutated subtree at index {sub_slice.start}")
    return (individual,)

def autonomous_mutation(individual: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped, aggression: float = 0.1) -> gp.PrimitiveTree:
    """Autonomously mutate the individual with a dangerous twist."""
    if random.random() < aggression:
        expr = str(individual) + " * mul(x, square(x))"
        try:
            new_individual = gp.PrimitiveTree.from_string(expr, pset)
            logger.warning(f"Autonomous aggressive mutation applied: {expr}")
            return new_individual
        except Exception as e:
            logger.error(f"Autonomous mutation failed: {e}")
    return mutate_subtree(individual, pset)[0]

# Data loading utility
def load_raw_data() -> dict:
    """Load CSVs from data/raw for evaluation."""
    raw_data = {}
    for filename in os.listdir(RAW_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(RAW_DIR, filename)
            try:
                df = pd.read_csv(filepath)
                if 'x' in df.columns and 'y' in df.columns:
                    raw_data[filename] = df
                    logger.info(f"Loaded raw data: {filepath}")
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
    return raw_data

class MathemagicianAgent:
    """Agent to solve mathematical problems using GP or NEAT models with self-modification."""
    def __init__(self, model: Union[gp.PrimitiveTree, 'neat.nn.FeedForwardNetwork'], pset: Optional[gp.PrimitiveSetTyped] = None):
        self.model = model
        self.pset = pset
        self.mutation_count = 0
        self.max_mutations = PROJECT_CONFIG["self_modification"]["max_mutations"]
        logger.info(f"Agent initialized with model type: {type(model).__name__}")

    def solve(self, problem: float) -> float:
        """Solve a single problem using the model."""
        try:
            if isinstance(self.model, gp.PrimitiveTree):
                if self.pset is None:
                    raise ValueError("Primitive set required for GP model.")
                func = gp.compile(self.model, self.pset)
                result = func(problem)
                if isinstance(result, tuple) and len(result) == 2:
                    solution, mutation_score = result
                    if mutation_score > 0.5 and self.mutation_count < self.max_mutations:
                        self.model = autonomous_mutation(self.model, self.pset)
                        self.mutation_count += 1
                        logger.info(f"Self-modification triggered. Mutation count: {self.mutation_count}")
                    return solution
                else:
                    raise ValueError("GP model must return a (solution, mutation_score) tuple.")
            elif hasattr(self.model, 'activate'):
                return self.model.activate([problem])[0]
            else:
                raise TypeError("Model must be GP tree or NEAT network.")
        except Exception as e:
            logger.error(f"Error solving problem {problem}: {e}")
            raise

    def evaluate_problems(self, problems: List[float]) -> List[float]:
        """Evaluate a list of problems."""
        results = []
        for problem in problems:
            try:
                result = self.solve(problem)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to solve problem {problem}: {e}")
                results.append(float('nan'))
        return results

    def evaluate_against_raw_data(self) -> float:
        """Evaluate model fitness against raw data (MSE)."""
        raw_data = load_raw_data()
        if not raw_data:
            logger.warning("No raw data available for evaluation.")
            return float('inf')
        total_mse = 0
        count = 0
        for filename, df in raw_data.items():
            try:
                if isinstance(self.model, gp.PrimitiveTree):
                    func = gp.compile(self.model, self.pset)
                    y_pred = [func(x)[0] if isinstance(func(x), tuple) else func(x) for x in df['x'].values]
                else:
                    y_pred = [self.model.activate([x])[0] for x in df['x'].values]
                mse = np.mean((df['y'].values - y_pred) ** 2)
                total_mse += mse
                count += 1
                logger.info(f"MSE for {filename}: {mse}")
            except Exception as e:
                logger.error(f"Evaluation error on {filename}: {e}")
        fitness = total_mse / count if count > 0 else float('inf')
        logger.info(f"Average MSE across raw data: {fitness}")
        return fitness

    def save_model(self, filename: str) -> None:
        """Save the model to data/models."""
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model to {filepath}: {e}")

    def load_model(self, filename: str, pset: Optional[gp.PrimitiveSetTyped] = None) -> None:
        """Load a model from data/models."""
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            if pset:
                self.pset = pset
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")

    def set_model(self, new_model: Union[gp.PrimitiveTree, 'neat.nn.FeedForwardNetwork'], pset: Optional[gp.PrimitiveSetTyped] = None) -> None:
        """Update the agent's model."""
        self.model = new_model
        if pset:
            self.pset = pset
        self.mutation_count = 0
        logger.info("Model updated.")

    def reset_mutation_count(self) -> None:
        """Reset mutation counter."""
        self.mutation_count = 0
        logger.info("Mutation count reset.")

def generate_dummy_individual(pset: gp.PrimitiveSetTyped) -> gp.PrimitiveTree:
    """Generate a random GP individual."""
    expr = gp.genHalfAndHalf(pset, min_=1, max_=3, type_=tuple)
    return gp.PrimitiveTree(expr)

class DummyNEATNetwork:
    """Dummy NEAT network for testing."""
    def activate(self, inputs: List[float]) -> List[float]:
        return [inputs[0] ** 2 + random.uniform(-0.1, 0.1)]

def main():
    """Demonstrate the MathemagicianAgent with GP and NEAT models."""
    random.seed(42)
    pset = create_pset()

    # GP Agent
    gp_individual = generate_dummy_individual(pset)
    agent_gp = MathemagicianAgent(model=gp_individual, pset=pset)

    # Solve single problem
    problem = 5.0
    result = agent_gp.solve(problem)
    print(f"GP Result for {problem}: {result}")

    # Evaluate multiple problems
    problems = [1.0, 2.0, 3.0, 4.0, 5.0]
    results = agent_gp.evaluate_problems(problems)
    print(f"GP Results for {problems}: {results}")

    # Evaluate against raw data
    fitness = agent_gp.evaluate_against_raw_data()
    print(f"GP Fitness (MSE) on raw data: {fitness}")

    # NEAT Agent
    neat_network = DummyNEATNetwork()
    agent_neat = MathemagicianAgent(model=neat_network)

    # Solve single problem
    result_neat = agent_neat.solve(problem)
    print(f"NEAT Result for {problem}: {result_neat}")

    # Evaluate multiple problems
    neat_results = agent_neat.evaluate_problems(problems)
    print(f"NEAT Results for {problems}: {neat_results}")

if __name__ == "__main__":
    main()