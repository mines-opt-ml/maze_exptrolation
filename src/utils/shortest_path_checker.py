import torch

def _shortest_path_checker(input, prediction, solution, device='cpu', verbose = False):
    '''
    Checks that the predicted shortest path, encoded as an RGB image in prediction, is indeed correct.
    Method: check that the length of the predicted path matches the length of the solution path,
    and that the predicted path contains the start and end points.
    '''
    # find number of pixels equal to 1 in the prediction
    num_pixels = torch.sum(prediction).item()
    # find number of pixels equal to 1 in the solution
    num_pixels_solution = torch.sum(solution).item()
    # check both pixel counts are divisible by 4, as they should be
    assert num_pixels % 4 == 0, f"Number of pixels in prediction is not divisible by 4: {num_pixels}"
    assert num_pixels_solution % 4 == 0, f"Number of pixels in solution is not divisible by 4: {num_pixels_solution}"
    # check that the number of pixels in the prediction is equal to the number of pixels in the solution
    length_equal = (num_pixels == num_pixels_solution)
    # Now check if prediction contains start and end points
    masked_input = input * prediction
    red_pixel = torch.tensor([1.0,0,0], device=device)
    green_pixel = torch.tensor([0,1.0,0], device=device)
    matches_start = torch.all(masked_input == red_pixel.view(3, 1, 1), dim=0)
    matches_end = torch.all(masked_input == green_pixel.view(3, 1, 1), dim=0)
    contains_start = torch.any(matches_start)
    contains_end = torch.any(matches_end)
    if verbose:
         print(f"Is length equal: {length_equal}")
         print(f"Contains start: {contains_start}")
         print(f"Contains end: {contains_end}")

    if length_equal & contains_start & contains_end:
        return 1.0
    else:
        return 0.0
    
def shortest_path_checker(input, prediction, solution):
    batch_size = input.shape[0]
    device = input.device
    results = []
    for i in range(batch_size):
        result = _shortest_path_checker(input[i], prediction[i], solution[i], device)
        results.append(result)
    results = torch.tensor(results, device=device)
    total_score = torch.sum(results) / batch_size
    return total_score, results