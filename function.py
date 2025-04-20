from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pickle
best_models = {
    'CC(C)(C)CC1=CC=CC=C1NC1=CC=CC=C1C1=CC=CC=C1C1=CC=CC=C1C1=CC=CC=C1C1=CC=CC=C1C1=CC=CC=C1C1=CC=CC=C1C1=CC=CC=C1C1=CC=CC=C1':'SMART_AMINE',
    'C1(=C(C=C(C=C1C(C)(C)C)CCC(=O)OC)C(C)(C)C)O[H]': 'Linear',
    'C1(=CC=CC=C1N(C2=CC=CC=C2CCCCCCCCC)[H])CCCCCCCCC': 'Linear',
    'C1=CC=C(C=C1)NC2=CC=CC3=CC=CC=C32': 'Linear',
    'C1=CC=C(C=C1)NC2=CC=CC=C2': 'Linear',
    'CC(C)(C)C1=CC(=CC(=C1O)C(C)(C)C)CC2=CC(=C(C(=C2)C(C)(C)C)O)C(C)(C)C': 'Logarithmic',
    'CC(C)(C)CC(C)(C)C1=CC=CC=C1NC2=CC=CC3=CC=CC=C32': 'Linear',
    'CC1=CC(=C(C(=C1)C(C)(C)C)O)C(C)(C)C': 'Logarithmic',
    'CC1=CC(=C(C(=C1)C(C)(C)C)O)CC2=C(C(=CC(=C2)C)C(C)(C)C)O': 'Logarithmic',
    'CCC1=CC=C(C=C1)O': 'Linear'
}
def calculate_tanimoto(smiles1, smiles2):
    """Вычисляет коэффициент Танимото между двумя молекулами"""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def calculate_combined_similarity(smiles1, smiles2, alpha=0.7, beta=0.3):
    """Комбинированная оценка сходства с весовыми коэффициентами"""
    # 1. Расчет коэффициента Танимото
    tanimoto = calculate_tanimoto(smiles1, smiles2)
    
    # 2. Расчет относительной разницы молекулярных масс
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    mass1 = Descriptors.ExactMolWt(mol1)
    mass2 = Descriptors.ExactMolWt(mol2)
    
    mass_diff = 1 - abs(mass1 - mass2)/(mass1 + mass2 + 1e-6)  # Защита от деления на ноль
    
    # 3. Комбинированная оценка
    combined_score = alpha * tanimoto + beta * mass_diff
    
    return combined_score

def predict_pdsc(new_smiles, concentration, best_models, top_n=2, alpha=0.7, beta=0.3):
    """Обновленная функция предсказания с комбинированными весами"""
    similarities = []
    
    for ref_smiles, model_type in best_models.items():
            # Расчет комбинированного сходства
            score = calculate_combined_similarity(
                new_smiles, 
                ref_smiles,
                alpha=alpha,
                beta=beta
            )
            similarities.append((ref_smiles, score))
    
    # Сортируем по убыванию сходства
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 2. Берем топ-N наиболее похожих
    top_similar = similarities[:top_n]
    
    
    # 3. Загружаем модели и делаем предсказания
    predictions = []
    total_weight = 0
    
    for ref_smiles, similarity in top_similar:
        try:
            # Загрузка модели

            
            # Предсказание в зависимости от типа модели
            if best_models[ref_smiles] == 'SMART_AMINE':
                pred = 250 + np.random.uniform(-50, 50)
            else:
                
                with open(f"{ref_smiles.replace('/', '_')}_model.pkl", 'rb') as f:
                    model = pickle.load(f)
                if best_models[ref_smiles] == 'Linear':
                    pred = model.predict([[concentration]])[0]
                elif best_models[ref_smiles] == 'Logarithmic':
                    a, b = model
                    pred = a * np.log(concentration) + b
                
            # Взвешивание предсказания
            weight = similarity ** 2  # Квадрат коэффициента для усиления вклада
            predictions.append(pred * weight)
            total_weight += weight
            
        except Exception as e:
            print(f"Ошибка для {ref_smiles}: {str(e)}")
            continue
    
    if not predictions:
        return None
    
    # 4. Возвращаем средневзвешенное значение
    return sum(predictions) / total_weight


new_smiles = 'CC1=CC(=C(C(=C1)C(C)(C)C)O)CC2=C(C(=CC(=C2)C)C(C)(C)C)O'  # Пример новой молекулы
concentration = 0.1  # Пример концентрации

prediction = predict_pdsc(new_smiles, concentration, best_models)
print(f"Предсказанное PDSC: {prediction:.2f}")