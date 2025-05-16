import { AIEngine } from './engine/AIEngine';
import { SchanuelConjecture } from './conjectures/implementations/SchanuelConjecture';

const engine = new AIEngine([
    new SchanuelConjecture(),
]);

const result = engine.analyze([1, 2, 3, 4, 5]);
console.log(`Analysis result: ${result}`);
