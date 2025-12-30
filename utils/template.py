def template_caption(inference_class, version="Gianni"):
    if version == "Gianni":
        return f"Describe the image of the subject {inference_class} in detail."
    if version == "Zihan":
        return """Describe the image, explain why we can classify it to a specific class, what are the features the most important. Only focus on its shape, do not describe and analyze its color. """
    return None


def preprompt_rating(version):
    if version == "short_cats_dogs_cars":
        text = """
        From an image caption, rate as a note from 1 to 5, how the description is good. Here is some examples.
        
        label: dog
        explanation: This explanation focuses on a face-like area with features such as a nose and eyes and some hair on either side of the cheeks, more consistent with a dog. 
        answer: 3

        label: cat
        explanation: This explanation focuses on the whole-body part of the animal, in which we can observe its hair and body shape, which is more in line with the characteristics of cats
        answer: 4

        label: cat
        explanation: The features in the image indicate that this is a cat.
        answer: 1

        label: car
        explanation: This interpretation focuses on the overall contours of the object and features wheels, making it very much in line with being a car. The overall silhouette is also more evidence that it is a car.
        answer: 4

        label: cat
        explanation: The image shows a cat lying on a chair, with its eyes glowing red. The cat is positioned on a red cushion, which adds a contrasting element to the scene. The cat's eyes are illuminated by a light source, making them appear red. This effect can be achieved by using a flash or a light source to create a reflection on the cat's eyes. The cat's position on the chair and its glowing eyes are what make it identifiable as a cat.
        answer: 4
   
        label: cat     
        explanation: This interpretation focuses on the cat's face, whose pointed ears and furry features indicate that it is most likely a cat
        answer: 3

        label: dog
        explanation: The image features a large brown dog standing in a room, with its head turned to the side. The dog appears to be looking at something or someone, possibly its owner. The dog's posture and size suggest that it is a canine, and its presence in the room indicates that it is a domesticated pet. The dog's position and the context of the image make it clear that it is a dog, and not a cat or another animal.
        answer: 4

        label: dog
        explanation: This picture is clearly an animal and fits the profile of a dog
        answer: 2
        
        Answer consisely, by outputing a number between 1 and 5. Just say a number.
        """
    elif version == "basic":
        """
        Return a number;"""
    return text


def template_rating(inference_class, description, version="Remi"):
    if version == "Remi":
        return f"""
            label: {inference_class} 
            explanation: {description}
            answer: """

    if version == "Zihan":
        return f"""
            Based on the EXPLAINATION and its scores that I have previously provided to you, rate the EXPLAINATION that I have provided below, giving only the scores and no other explanations, with no numbers appearing other than the scores: 
            {description}"""

    return None
