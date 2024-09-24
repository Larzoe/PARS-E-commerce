from locust import HttpUser, TaskSet, task, between
import random 
from time import time

events = ['category_banner_view', 'inspiratie_view',
          'advies_view', 'mijnsanitair_view', 'view_item_list', 'view_item', 'add_to_cart', 
          'remove_from_cart', 'begin_checkout', 'view_cart', 'add_to_wishlist', 'view_search_results',
          'product_compare']

class UserBehavior(TaskSet):
    @task
    def send_event(self):
        self.client.post("/events", json={
            "ga_stream_cookie": f"{random.randint(0,50)}",
            "event": events[random.randint(0,len(events)-1)],
            "timestamp": f"{round(time()*1000)}"
        })

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)
